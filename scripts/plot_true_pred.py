"""Render true-vs-predicted vacuity scatter plots for the two all-data models.

Loads the pickled ``all_data`` and ``all_data_tagged`` models, predicts on a fixed
1000-row sample, and writes linear / log-log / zoomed scatter plots. Paths come
from config.
"""

import os
import pickle

import pandas as pd

from stratml import config
from stratml.modeling.plotting import plot_scatter

ALL_DIR = config.PLOTS_DIR / "all_data"
TAGGED_DIR = config.PLOTS_DIR / "all_tagged_data"


def _load_sample():
    full_data = pd.read_csv(config.LAYER_STATS_CSV)
    full_data = full_data.dropna(subset=['High_Erosion'])
    full_data = full_data[full_data['Marine'] != 1]
    full_data = full_data[full_data['Layer_Thickness'] >= config.MIN_THICKNESS]
    return full_data.sample(n=1_000, random_state=config.RANDOM_STATE).copy()


def _predict(model_path, data_sampled, features):
    with open(model_path, 'rb') as f:
        saved_model = pickle.load(f)
        model = saved_model['model']
        x_scaler = saved_model['x_scaler']
        y_scaler = saved_model['y_scaler']

    data = data_sampled.copy()
    x = data[features].copy()
    y = data[config.TARGETS].copy()

    # Normalize inputs
    x.loc[:, 'Layer_Thickness'] /= config.THICKNESS_SCALE
    x.loc[:, 'Layer_Time'] /= config.TIME_SCALE

    # Predict
    x_scaled_values = x_scaler.transform(x)
    y_pred_scaled = model.predict(x_scaled_values)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_pred = pd.DataFrame(y_pred, columns=y.columns)

    # Calculate vacuity metrics
    true_dep = y['Total_Dep'].values / config.THICKNESS_SCALE - x['Layer_Thickness'].values
    pred_dep = y_pred['Total_Dep'].values - x['Layer_Thickness'].values
    true_time = y['Total_Time'].values / config.TIME_SCALE - x['Layer_Time'].values
    pred_time = y_pred['Total_Time'].values - x['Layer_Time'].values
    return true_dep, pred_dep, true_time, pred_time


def _make_plots(out_dir, true_dep, pred_dep, true_time, pred_time,
                dep_loglog_xlabel):
    # dep_loglog_xlabel differs per model: untagged 'log($D_{T}$)', tagged '$D_{T}$'
    os.makedirs(out_dir, exist_ok=True)

    plot_scatter(true_dep, pred_dep, out_dir / "Total_Dep_Scatter.png",
                 "Depositional Vacuity: Predicted vs. Observed", (0, 3), (0, 3),
                 xlabel=r'$D_{T}$', ylabel=r'$D_{P}$')

    plot_scatter(true_dep, pred_dep, out_dir / "Total_Dep_Scatter_loglog.png",
                 "Depositional Vacuity: Predicted vs. Observed", (1e-3, 4), (1e-3, 4),
                 xlabel=dep_loglog_xlabel, ylabel=r'$D_{P}$', xscale='log', yscale='log')

    plot_scatter(true_time, pred_time, out_dir / "Total_Time_Scatter.png",
                 "Temporal Vacuity: Predicted vs. Observed", (0, 1), (0, 1),
                 xlabel=r'$T_{T}$', ylabel=r'$T_{P}$')

    plot_scatter(true_time, pred_time, out_dir / "Total_Time_Scatter_loglog.png",
                 "Temporal Vacuity: Predicted vs. Observed", (1e-3, 1), (1e-3, 1),
                 xlabel=r'$T_{T}$', ylabel=r'$T_{P}$', xscale='log', yscale='log')

    plot_scatter(true_time, pred_time, out_dir / "Total_Time_Scatter_Zoomed.png",
                 "Temporal Vacuity: Predicted vs. Observed (Zoomed)", (0, 0.2), (0, 0.2),
                 xlabel=r'$T_{T}$', ylabel=r'$T_{P}$')

    plot_scatter(true_time, pred_time, out_dir / "Total_Time_Scatter_Zoomed_loglog.png",
                 "Temporal Vacuity: Predicted vs. Observed (Zoomed log-log)",
                 (1e-3, 0.2), (1e-3, 0.2), xlabel=r'$T_{T}$', ylabel=r'$T_{P}$',
                 xscale='log', yscale='log')


def main():
    data_sampled = _load_sample()

    # Model 1: untagged features
    _make_plots(ALL_DIR, *_predict(
        config.SAVED_MODELS_DIR / 'all_data_model.pkl',
        data_sampled, config.FEATURES_ALL),
        dep_loglog_xlabel=r'log($D_{T}$)')

    # Model 2: tagged features
    _make_plots(TAGGED_DIR, *_predict(
        config.SAVED_MODELS_DIR / 'all_data_tagged_model.pkl',
        data_sampled, config.FEATURES_ALL_TAGGED),
        dep_loglog_xlabel=r'$D_{T}$')


if __name__ == "__main__":
    main()
