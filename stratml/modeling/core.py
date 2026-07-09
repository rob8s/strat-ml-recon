"""Shared random-forest training/evaluation helpers.

Holds what the all-data trainers share: building the multi-output random forest,
reporting MAE/R2, and the full train routine (``train_all``). The two entry
scripts differ only in the feature set, sample size, and output filename.
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from stratml import config
from stratml.config import RANDOM_STATE


def build_model(params, random_state=RANDOM_STATE):
    """Multi-output random forest regressor."""
    return MultiOutputRegressor(
        RandomForestRegressor(random_state=random_state, **params)
    )


def report_metrics(y_true, y_pred, targets, indent="", blank_line_before_r2=True):
    """Compute and print per-target MAE and R2 (``multioutput='raw_values'``).

    ``indent`` and ``blank_line_before_r2`` control the console formatting.
    Returns ``(mae, r2)``.
    """
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")

    print("Mean Absolute Error for each target:")
    for target, error in zip(targets, mae):
        print(f"{indent}{target}: {error:.4f}")

    print(("\n" if blank_line_before_r2 else "") + "R-squared score for each target:")
    for target, score in zip(targets, r2):
        print(f"{indent}{target}: {score:.4f}")

    return mae, r2


def train_all(features, sample_n, model_filename):
    """Train one multi-output RF over all (non-marine) environments and pickle it.

    Shared body of ``train_all_data`` and ``train_all_data_tagged``.
    """
    # Load and preprocess data
    dfg = pd.read_csv(config.LAYER_STATS_CSV)

    # Filter data: exclude marine deposits and small layers
    dfg = dfg[dfg['Marine'] != 1]
    dfg = dfg[dfg['Layer_Thickness'] >= config.MIN_THICKNESS]

    # Convert data types
    dfg = dfg.astype({
        'Layer_Thickness': float,
        'Layer_Time': float,
        'Lobe': int,
        'Channel': int,
        'Wet_Floodplain': int,
        'Dry_Floodplain': int,
        'Marine': int,
        'Total_Dep': float,
        'Total_Time': float,
        'Stasis_Proportion': float,
        'Deposition_Proportion': float,
        'High_Erosion': int
    })

    # Normalize features and targets
    dfg['Layer_Thickness'] /= config.THICKNESS_SCALE
    dfg['Total_Dep'] /= config.THICKNESS_SCALE
    dfg['Layer_Time'] /= config.TIME_SCALE
    dfg['Total_Time'] /= config.TIME_SCALE

    # Sample data
    dfg_sampled = dfg.sample(n=sample_n, random_state=config.RANDOM_STATE)

    # Extract features and targets
    x = dfg_sampled[features]
    y = dfg_sampled[config.TARGETS]

    # Scale features and targets
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_scaled, test_size=0.2, random_state=config.RANDOM_STATE)

    # Train model
    multi_rf = build_model(config.RF_PARAMS_ALL)
    multi_rf.fit(x_train, y_train)

    # Predict and evaluate
    y_pred_scaled = multi_rf.predict(x_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_unscaled = y_scaler.inverse_transform(y_test)

    report_metrics(y_test_unscaled, y_pred, config.TARGETS)

    # Save model
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    model_file_path = os.path.join(config.SAVED_MODELS_DIR, model_filename)
    with open(model_file_path, 'wb') as f:
        pickle.dump({
            'model': multi_rf,
            'x_scaler': x_scaler,
            'y_scaler': y_scaler
        }, f)

    return multi_rf
