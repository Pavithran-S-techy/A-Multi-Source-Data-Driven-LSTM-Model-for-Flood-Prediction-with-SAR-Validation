import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


SEQ_LEN = 7


# Load regression datasets
train = pd.read_csv("train_regression.csv")
val   = pd.read_csv("validation_regression.csv")


features = [
    "rainfall_mm",
    "rain_3day",
    "rain_5day",
    "rain_7day",
    "extremerain",
    "elevation_m",
    "slope_deg"
]


X_train = train[features].values
y_train = train["flood_risk"].values

X_val = val[features].values
y_val = val["flood_risk"].values


# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

joblib.dump(scaler, "scaler_regression.pkl")


# Create sequences
def make_seq(X, y):

    Xs, ys = [], []

    for i in range(SEQ_LEN - 1, len(X)):
        Xs.append(X[i-SEQ_LEN+1:i+1])
        ys.append(y[i])

    return np.array(Xs), np.array(ys)


X_train, y_train = make_seq(X_train, y_train)
X_val, y_val     = make_seq(X_val, y_val)


# Build regression LSTM model
model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(SEQ_LEN,7)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(64))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32, activation="relu"))

# Regression output layer
model.add(Dense(1, activation="sigmoid"))


model.compile(
    optimizer=Adam(0.001),
    loss="mse",
    metrics=["mae"]
)


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "flood_regression_model.h5",
    monitor="val_loss",
    save_best_only=True
)


# Train model
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop, checkpoint],
    verbose=1
)


print("\nRegression flood risk model trained successfully.")
