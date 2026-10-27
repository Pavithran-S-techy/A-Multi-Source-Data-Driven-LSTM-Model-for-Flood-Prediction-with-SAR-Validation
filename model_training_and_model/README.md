
# LSTM-Based Flood Risk Prediction Model

## Overview
This folder contains the Python scripts and saved model files used for training, testing, and evaluating the LSTM-based flood risk prediction model developed as part of the project **A Multi-Source Data-Driven LSTM Model for Flood Prediction with SAR Validation**.

The model estimates a continuous flood risk score using rainfall and terrain-related features.

## Model Architecture
The model uses a Long Short-Term Memory (LSTM) neural network to process sequences of seven consecutive time steps.

**Input features:**
- Daily rainfall (`rainfall_mm`)
- 3-day accumulated rainfall (`rain_3day`)
- 5-day accumulated rainfall (`rain_5day`)
- 7-day accumulated rainfall (`rain_7day`)
- Extreme rainfall indicator (`extremerain`)
- Elevation (`elevation_m`)
- Slope (`slope_deg`)

The network includes two LSTM layers, batch normalization, dropout, and dense layers. It uses the Adam optimizer and mean squared error (MSE) as the loss function.

## Files Description

| File | Description |
|---|---|
| `train_regression_lstm.py` | Defines and trains the LSTM regression model. |
| `create_regression_dataset.py` | Prepares regression datasets and calculates continuous flood risk scores. |
| `evaluate_regression_model.py` | Evaluates model predictions using MAE, RMSE, and R². |
| `test_regression_model.py` | Tests predictions using low, medium, and high rainfall scenarios. |
| `test_regression_complex.py` | Tests additional scenarios involving rainfall and terrain variations. |
| `flood_regression_model.h5` | Saved trained model. |
| `scaler_regression.pkl` | Saved feature scaler used during preprocessing. |

## Requirements
The following Python libraries are used by the scripts:

- NumPy
- Pandas
- scikit-learn
- TensorFlow / Keras
- Joblib

Install the dependencies using:

```bash
pip install numpy pandas scikit-learn tensorflow joblib
```

## Training the Model
Ensure that the required training and validation CSV files are available in the working directory.

Run:

```bash
python train_regression_lstm.py
```

The script scales the input features, creates seven-step sequences, trains the LSTM model, and saves the best-performing model as `flood_regression_model.h5`. The fitted scaler is saved as `scaler_regression.pkl`.

## Dataset Preparation
If the required source datasets are available, run:

```bash
python create_regression_dataset.py
```

This script reads the source training, validation, and testing datasets and generates the corresponding regression datasets.

## Model Evaluation
To evaluate the trained model, ensure that `test_regression.csv`, `flood_regression_model.h5`, and `scaler_regression.pkl` are available, then run:

```bash
python evaluate_regression_model.py
```

The evaluation script reports:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

## Testing
Run the following scripts to test predefined rainfall and terrain scenarios:

```bash
python test_regression_model.py
python test_regression_complex.py
```

These scripts display predicted flood risk scores and map them to GREEN, YELLOW, or RED alert categories using predefined thresholds.

## Notes
- The required CSV datasets must be available before running the corresponding scripts.
- The saved model and scaler are required for evaluation and scenario testing.
- Ensure that the input features follow the expected order and that the input sequences contain seven time steps.
- This folder contains the model training and evaluation components of the project.
