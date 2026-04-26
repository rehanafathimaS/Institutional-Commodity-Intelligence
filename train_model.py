import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 1. LOAD AND PREPARE DATA
df = pd.read_csv('wb_commodity_price_intelligence.CSV')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['commodity_name', 'date'])

# 2. ADVANCED FEATURE ENGINEERING
# Adding Lags and Moving Averages for better "Intelligence"
df['lag_1'] = df.groupby('commodity_name')['price_nominal_usd'].shift(1)
df['lag_2'] = df.groupby('commodity_name')['price_nominal_usd'].shift(2)
df['rolling_mean_3'] = df.groupby('commodity_name')['price_nominal_usd'].transform(lambda x: x.rolling(3).mean())

era_mapping = {
    'Pre-Oil Shock Era (pre-1970)': 1, 'Oil Shock & Volatility Era (1970s)': 2, 
    'Great Moderation & Globalization (1980s-1990s)': 3, 'Commodity Supercycle Era (2000s)': 4, 
    'Post-Crisis & Shale Revolution (2010s)': 5, 'COVID & Post-Pandemic Era (2020s)': 6
}
df['era_score'] = df['era'].map(era_mapping).fillna(0)

df_clean = df.dropna(subset=['lag_1', 'lag_2', 'rolling_mean_3'])

# 3. SCALER (Fixed for app.py compatibility)
scaler = StandardScaler()
scaler.fit(df_clean['price_nominal_usd'].values.reshape(-1, 1))

# 4. TRAIN MODEL
X = df_clean[['lag_1', 'lag_2', 'rolling_mean_3', 'era_score', 'category']]
X = pd.get_dummies(X, columns=['category'], prefix='cat')
y = df_clean['price_nominal_usd']

features = X.columns.tolist()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. SAVE ASSETS
with open('commodity_model.pkl', 'wb') as f: pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('features.pkl', 'wb') as f: pickle.dump(features, f)

print("✅ Professional Model Assets Created!")