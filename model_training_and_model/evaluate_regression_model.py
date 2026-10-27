import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load model and scaler
model = load_model("flood_regression_model.h5", compile=False)
scaler = joblib.load("scaler_regression.pkl")

# Load test dataset
df = pd.read_csv("test_regression.csv")

features = [
    "rainfall_mm",
    "rain_3day",
    "rain_5day",
    "rain_7day",
    "extremerain",
    "elevation_m",
    "slope_deg"
]

X = df[features].values
y_true = df["flood_risk"].values


# Scale
X_scaled = scaler.transform(X)

# Create sequences
SEQ_LEN = 7

X_seq = []
y_seq = []

for i in range(SEQ_LEN - 1, len(X_scaled)):
    X_seq.append(X_scaled[i-SEQ_LEN+1:i+1])
    y_seq.append(y_true[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)


# Predict
y_pred = model.predict(X_seq).flatten()


# Calculate scores
mae = mean_absolute_error(y_seq, y_pred)
rmse = np.sqrt(mean_squared_error(y_seq, y_pred))
r2 = r2_score(y_seq, y_pred)


print("\nMODEL PERFORMANCE")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)
