import pandas as pd
import numpy as np


# Load datasets
train = pd.read_csv("train_smote.csv")
val   = pd.read_csv("validation.csv")
test  = pd.read_csv("test.csv")


combined = pd.concat([train, val, test], ignore_index=True)


# Normalize factors
rain_norm = combined["rain_7day"] / combined["rain_7day"].max()

elev_norm = 1 - (combined["elevation_m"] / combined["elevation_m"].max())

slope_norm = 1 - (combined["slope_deg"] / combined["slope_deg"].max())


# Create continuous flood risk score
combined["flood_risk"] = (
    rain_norm**2
    * (1 + 0.4 * elev_norm)
    * (1 + 0.3 * slope_norm)
)

combined["flood_risk"] = combined["flood_risk"].clip(0, 1)


# Split back
train_new = combined.iloc[:len(train)]
val_new   = combined.iloc[len(train):len(train)+len(val)]
test_new  = combined.iloc[len(train)+len(val):]


# Save new datasets
train_new.to_csv("train_regression.csv", index=False)
val_new.to_csv("validation_regression.csv", index=False)
test_new.to_csv("test_regression.csv", index=False)


print("Regression datasets created successfully.")
