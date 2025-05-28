import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (adjust filename if needed)
df = pd.read_csv("household_power_consumption.txt", sep=';', na_values='?', low_memory=False)

# Combine Date and Time into Datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
df.drop(columns=['Date', 'Time'], inplace=True)

# Drop missing values and convert to float
df.dropna(inplace=True)
for col in df.columns:
    if col != 'Datetime':
        df[col] = df[col].astype(float)

# Feature Engineering
df['Hour'] = df['Datetime'].dt.hour
df['Day'] = df['Datetime'].dt.day
df['Month'] = df['Datetime'].dt.month
df['DayOfWeek'] = df['Datetime'].dt.dayofweek

# Resample to hourly data (use lowercase 'h' instead of 'H')
df_hourly = df.resample('h', on='Datetime').mean().reset_index()

# Drop rows with NaN in target after resampling
df_hourly = df_hourly.dropna(subset=['Global_active_power'])

# Prepare X and y
X = df_hourly.drop(columns=['Datetime', 'Global_active_power'])
y = df_hourly['Global_active_power']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test_scaled)

# Calculate RMSE manually since 'squared' parameter is not supported in your version
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 RMSE: {rmse:.3f}")
print(f"📊 MAE: {mae:.3f}")
print(f"📊 R²: {r2:.3f}")

# Feature Importance
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

# Plot without palette and emoji to avoid warnings
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()
