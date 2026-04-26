import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. PAGE SETUP
st.set_page_config(page_title="Institutional Commodity Intel", layout="wide")

# 2. ADVANCED CSS (Dark Theme & Professional Cards)
st.markdown("""
<style>
    .main { background-color: #0d1117; color: white; }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: 0.3s;
    }
    .metric-card:hover { border-color: #58a6ff; }
    .metric-value { font-size: 26px; font-weight: bold; color: #ffffff; margin-bottom: 5px; }
    .metric-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# 3. ASSETS LOADING
@st.cache_resource
def load_ml_assets():
    with open('commodity_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('features.pkl', 'rb') as f: features = pickle.load(f)
    return model, scaler, features

@st.cache_data
def load_market_data():
    df = pd.read_csv('wb_commodity_price_intelligence.CSV')
    df['date'] = pd.to_datetime(df['date'])
    return df

model, scaler, features = load_ml_assets()
df = load_market_data()

# --- 4. SIDEBAR: PROFESSIONAL INPUT PANEL ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
st.sidebar.header("🕹️ Analysis Controls")

st.sidebar.subheader("📅 Time Horizon")
selected_date = st.sidebar.date_input("Target Date", value=df['date'].max())
year_input = st.sidebar.number_input("Analysis Year", 1960, 2026, int(selected_date.year))

st.sidebar.subheader("📦 Asset Intelligence")
cat_list = sorted(df['category'].unique())
selected_cat = st.sidebar.selectbox("Market Category", cat_list)

comm_list = sorted(df[df['category'] == selected_cat]['commodity_name'].unique())
selected_comm = st.sidebar.selectbox("Commodity Asset", comm_list)

units = df[df['commodity_name'] == selected_comm]['unit'].unique()
st.sidebar.selectbox("Trading Unit", units)

# --- 5. DATA LOGIC ---
comm_df = df[df['commodity_name'] == selected_comm].sort_values('date')
current_price = comm_df['price_nominal_usd'].iloc[-1]

# --- FIXED PREDICTION LOGIC ---
# Get last 3 prices for lagging
last_3_prices = comm_df['price_nominal_usd'].tail(3).values

# 1. Create a DataFrame with the exact same 9 columns used in training
input_df = pd.DataFrame(columns=features)
input_df.loc[0] = 0  # Initialize everything with 0

# 2. Fill the lagging features
input_df['lag_1'] = last_3_prices[-1]
input_df['lag_2'] = last_3_prices[-2]
input_df['rolling_mean_3'] = np.mean(last_3_prices)

# 3. Fill the Category dummy (One-hot encoding)
cat_col = f"cat_{selected_cat}" 
if cat_col in input_df.columns:
    input_df[cat_col] = 1

# 4. Scale and Predict (Variable name changed to pred_price to match card)
input_scaled = scaler.transform(input_df)
pred_price = model.predict(input_scaled)[0]

# Confidence Logic
volatility = comm_df['price_nominal_usd'].tail(12).std() / comm_df['price_nominal_usd'].tail(12).mean()
conf_level = "HIGH" if volatility < 0.12 else "MODERATE" if volatility < 0.22 else "LOW"
conf_color = "#3fb950" if conf_level == "HIGH" else "#d29922" if conf_level == "MODERATE" else "#f85149"

# --- 6. MAIN DASHBOARD ---
st.title("🌐 Institutional Commodity Intelligence")
st.caption(f"Real-time analysis for {selected_comm} | Data Coverage: 1960 - 2026")

# KPI TILES
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Records</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="metric-card"><div class="metric-label">Market Price</div><div class="metric-value">${current_price:,.2f}</div></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="metric-card"><div class="metric-label">Forecasted</div><div class="metric-value" style="color:#58a6ff">${pred_price:,.2f}</div></div>', unsafe_allow_html=True)
with k4: st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value" style="color:{conf_color}">{conf_level}</div></div>', unsafe_allow_html=True)
with k5: 
    outlook = "BULLISH" if pred_price > current_price else "BEARISH"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Outlook</div><div class="metric-value" style="color:{"#3fb950" if outlook=="BULLISH" else "#f85149"}">{outlook}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# PRIMARY CHARTS
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📈 Historical Price Trend & Forecast Area")
    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(x=comm_df['date'], y=comm_df['price_nominal_usd'], fill='tozeroy', name='Actual Price', line=dict(color='#3fb950', width=2), fillcolor='rgba(63, 185, 80, 0.1)'))
    
    # Add Forecast Marker
    next_mo = comm_df['date'].max() + pd.DateOffset(months=1)
    fig_area.add_trace(go.Scatter(x=[next_mo], y=[pred_price], mode='markers+text', text=["Prediction"], textposition="top center", marker=dict(size=12, color='#f85149', symbol='diamond'), name='Forecast'))
    
    fig_area.update_layout(template="plotly_dark", hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig_area, use_container_width=True)

with c2:
    st.subheader(f"🍩 {selected_cat} Internal Share")
    filtered_cat_df = df[df['category'] == selected_cat]
    donut_data = filtered_cat_df.groupby('commodity_name')['price_nominal_usd'].mean().sort_values(ascending=False).head(10).reset_index()
    
    fig_donut = px.pie(donut_data, values='price_nominal_usd', names='commodity_name', hole=0.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.Greens_r)
    fig_donut.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20), annotations=[dict(text='Price<br>Share', x=0.5, y=0.5, font_size=14, showarrow=False)])
    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_donut, use_container_width=True)

# EXTRA ANALYSIS
st.markdown("---")
b1, b2 = st.columns(2)

with b1:
    st.subheader(f"🏆 Top 10 Price Leaders: {selected_cat}")
    top_10 = df[df['category'] == selected_cat].groupby('commodity_name')['price_nominal_usd'].mean().sort_values(ascending=False).head(10).reset_index()
    fig_bar = px.bar(top_10, x='price_nominal_usd', y='commodity_name', orientation='h', template="plotly_dark", color='price_nominal_usd', color_continuous_scale='Viridis')
    st.plotly_chart(fig_bar, use_container_width=True)

with b2:
    st.subheader("📅 Seasonal Monthly Averages")
    comm_df['month'] = comm_df['date'].dt.month_name()
    m_avg = comm_df.groupby('month')['price_nominal_usd'].mean().reindex(['January','February','March','April','May','June','July','August','September','October','November','December']).reset_index()
    fig_m = px.line(m_avg, x='month', y='price_nominal_usd', markers=True, template="plotly_dark", color_discrete_sequence=['#58a6ff'])
    st.plotly_chart(fig_m, use_container_width=True)

st.subheader("📋 Recent Market Data Explorer")
st.dataframe(comm_df[['date', 'price_nominal_usd', 'unit']].tail(15).sort_values('date', ascending=False), use_container_width=True)