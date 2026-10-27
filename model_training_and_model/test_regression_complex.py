import numpy as np
import joblib
from tensorflow.keras.models import load_model


model = load_model("flood_regression_model.h5", compile=False)
scaler = joblib.load("scaler_regression.pkl")

SEQ_LEN = 7


def classify(risk):

    if risk < 0.08:
        return "GREEN"
    elif risk < 0.35:
        return "YELLOW"
    else:
        return "RED"


def test_case(input_data, name):

    input_scaled = scaler.transform(input_data)
    input_scaled = input_scaled.reshape(1, SEQ_LEN, 7)

    risk = float(model.predict(input_scaled, verbose=0)[0][0])

    print("\n==============================")
    print("Scenario:", name)
    print("Flood Risk Score:", round(risk, 4))
    print("Alert:", classify(risk))


# Scenario 1: High rainfall but HIGH elevation (mountain region)
mountain_heavy_rain = np.array([
    [120, 200, 300, 400, 1, 900, 15],
    [130, 220, 320, 420, 1, 900, 15],
    [140, 240, 350, 450, 1, 900, 15],
    [150, 260, 380, 480, 1, 900, 15],
    [160, 280, 410, 510, 1, 900, 15],
    [170, 300, 440, 540, 1, 900, 15],
    [180, 320, 470, 570, 1, 900, 15]
])


# Scenario 2: Moderate rainfall but VERY LOW elevation (floodplain)
floodplain_moderate = np.array([
    [30, 60, 90, 120, 0, 5, 1],
    [35, 70, 105, 140, 0, 5, 1],
    [40, 80, 120, 160, 0, 5, 1],
    [45, 90, 135, 180, 0, 5, 1],
    [50, 100, 150, 200, 0, 5, 1],
    [55, 110, 165, 220, 0, 5, 1],
    [60, 120, 180, 240, 0, 5, 1]
])


# Scenario 3: Mixed rainfall pattern
mixed_pattern = np.array([
    [10, 20, 40, 60, 0, 100, 4],
    [50, 80, 110, 140, 0, 100, 4],
    [20, 40, 60, 80, 0, 100, 4],
    [90, 150, 200, 260, 1, 100, 4],
    [30, 50, 80, 110, 0, 100, 4],
    [110, 180, 240, 310, 1, 100, 4],
    [60, 100, 150, 210, 0, 100, 4]
])


test_case(mountain_heavy_rain, "Heavy rain in mountains")
test_case(floodplain_moderate, "Moderate rain in floodplain")
test_case(mixed_pattern, "Mixed rainfall pattern")
