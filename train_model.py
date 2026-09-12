import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

# 1. DATA LOADING
print("⏳ Loading data...")
df = pd.read_csv('wb_commodity_price_intelligence.CSV')
df['date'] = pd.to_datetime(df['date'])

# 2. FEATURE ENGINEERING (Simplified)
df = df.sort_values(['commodity_name', 'date'])
df['lag_1'] = df.groupby('commodity_name')['price_nominal_usd'].shift(1)
df['lag_2'] = df.groupby('commodity_name')['price_nominal_usd'].shift(2)
df['rolling_mean_3'] = df.groupby('commodity_name')['price_nominal_usd'].transform(lambda x: x.rolling(window=3).mean())

# Drop rows with NaN from lagging
df.dropna(inplace=True)

# 3. PREPARING FEATURES
# One-hot encoding for categories
df_final = pd.get_dummies(df, columns=['category'], prefix='cat')
features = ['lag_1', 'lag_2', 'rolling_mean_3'] + [col for col in df_final.columns if col.startswith('cat_')]
X = df_final[features]
y = df_final['price_nominal_usd']

# 4. SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. TRAINING LIGHTWEIGHT MODEL
print("🚀 Training lightweight model...")
# NOTE: n_estimators=50 and max_depth=10 will keep the .pkl file size very small!
model = RandomForestRegressor(
    n_estimators=50, 
    max_depth=10, 
    min_samples_leaf=4,
    random_state=42, 
    n_jobs=-1
)
model.fit(X_scaled, y)

# 6. SAVING ASSETS
print("💾 Saving assets...")
with open('commodity_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

print("✅ Success! Files 'commodity_model.pkl', 'scaler.pkl', and 'features.pkl' are created.")
