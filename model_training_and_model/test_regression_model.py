import numpy as np
import joblib
from tensorflow.keras.models import load_model


# Load regression model and scaler
model = load_model("flood_regression_model.h5", compile=False)
scaler = joblib.load("scaler_regression.pkl")

SEQ_LEN = 7


# Alert mapping
def classify(risk):

    if risk < 0.08:
        return "GREEN"
    elif risk < 0.35:
        return "YELLOW"
    else:
        return "RED"



# Test function
def test_case(input_data, name):

    input_scaled = scaler.transform(input_data)
    input_scaled = input_scaled.reshape(1, SEQ_LEN, 7)

    risk = float(model.predict(input_scaled, verbose=0)[0][0])

    alert = classify(risk)

    print("\n==========================")
    print("Test:", name)
    print("Flood Risk Score:", round(risk, 4))
    print("Alert:", alert)


# LOW rainfall
low = np.array([
    [2, 5, 10, 20, 0, 600, 12],
    [3, 6, 12, 22, 0, 600, 12],
    [2, 5, 11, 21, 0, 600, 12],
    [3, 7, 13, 24, 0, 600, 12],
    [4, 8, 15, 26, 0, 600, 12],
    [3, 7, 14, 25, 0, 600, 12],
    [2, 6, 12, 23, 0, 600, 12]
])


# MEDIUM rainfall
medium = np.array([
    [30, 60, 90, 130, 0, 200, 5],
    [35, 70, 105, 150, 0, 200, 5],
    [40, 80, 120, 170, 0, 200, 5],
    [45, 90, 135, 190, 0, 200, 5],
    [50, 100, 150, 210, 0, 200, 5],
    [55, 110, 165, 230, 0, 200, 5],
    [60, 120, 180, 250, 0, 200, 5]
])


# HIGH rainfall
high = np.array([
    [120, 240, 360, 500, 1, 10, 1],
    [130, 260, 390, 540, 1, 10, 1],
    [140, 280, 420, 580, 1, 10, 1],
    [150, 300, 450, 620, 1, 10, 1],
    [160, 320, 480, 660, 1, 10, 1],
    [170, 340, 510, 700, 1, 10, 1],
    [180, 360, 540, 740, 1, 10, 1]
])


# Run tests
test_case(low, "LOW rainfall")
test_case(medium, "MEDIUM rainfall")
test_case(high, "HIGH rainfall")
