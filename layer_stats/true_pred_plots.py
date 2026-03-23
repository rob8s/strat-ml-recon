import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load and filter data
csv_file_path = 'data/final_data/Layer_Stats_Env_Tagged.csv'
full_data = pd.read_csv(csv_file_path)

# Apply filtering
full_data = full_data.dropna(subset=['High_Erosion'])
full_data = full_data[full_data['Marine'] != 1]
full_data = full_data[full_data['Layer_Thickness'] >= 0.065]

# Sample for consistent comparison between models
data_sampled = full_data.sample(n=1_000, random_state=42).copy()


def plot_scatter(true_vals, pred_vals, outpath, title, xlim, ylim, xlabel, ylabel, 
                 xscale='linear', yscale='linear', eps=1e-8):
    """Create scatter plot comparing true vs predicted values.
    
    Args:
        true_vals: Array of true values
        pred_vals: Array of predicted values
        outpath: Output file path
        title: Plot title
        xlim: X-axis limits
        ylim: Y-axis limits
        xlabel: X-axis label
        ylabel: Y-axis label
        xscale: 'linear' or 'log'
        yscale: 'linear' or 'log'
        eps: Clipping value for log scales
    """
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    # Clip to positive values for log scales
    true_vals_plot = np.clip(true_vals, eps, None) if xscale == 'log' else true_vals
    pred_vals_plot = np.clip(pred_vals, eps, None) if yscale == 'log' else pred_vals

    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(true_vals_plot, pred_vals_plot, alpha=0.4, edgecolor='k', s=7)
    
    # Plot 1:1 reference line
    if xscale == 'log' and yscale == 'log':
        plt.plot(xlim, ylim, linestyle="--", color="red", linewidth=1, 
                label="Perfect Prediction (1:1)")
    else:
        plt.plot([0, max(xlim)], [0, max(ylim)], linestyle="--", color="red",
                linewidth=1, label="Perfect Prediction (1:1)")

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(True, linewidth=1, alpha=1, which='minor')
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


# Model 1: Untagged features

with open('saved_models/all_data_model.pkl', 'rb') as f:
    saved_model = pickle.load(f)
    model = saved_model['model']
    x_scaler = saved_model['x_scaler']
    y_scaler = saved_model['y_scaler']

# Prepare data
data = data_sampled.copy()
x = data[['Layer_Thickness', 'Layer_Time', 'Lobe', 'Channel',
          'Wet_Floodplain', 'Dry_Floodplain', 'Marine']].copy()
y = data[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']].copy()

# Normalize inputs
x.loc[:, 'Layer_Thickness'] /= 6.5
x.loc[:, 'Layer_Time'] /= 115

# Predict
x_scaled_values = x_scaler.transform(x)
y_pred_scaled = model.predict(x_scaled_values)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_pred = pd.DataFrame(y_pred, columns=y.columns)

# Calculate vacuity metrics
true_dep = y['Total_Dep'].values / 6.5 - x['Layer_Thickness'].values
pred_dep = y_pred['Total_Dep'].values - x['Layer_Thickness'].values
true_time = y['Total_Time'].values / 115 - x['Layer_Time'].values
pred_time = y_pred['Total_Time'].values - x['Layer_Time'].values

# Generate plots
plot_scatter(true_dep, pred_dep,
             "y_pred_plots/all_data/Total_Dep_Scatter.png",
             "Depositional Vacuity: Predicted vs. Observed", (0, 3), (0, 3), 
             xlabel=r'$D_{T}$', ylabel=r'$D_{P}$')

plot_scatter(true_dep, pred_dep,
             "y_pred_plots/all_data/Total_Dep_Scatter_loglog.png",
             "Depositional Vacuity: Predicted vs. Observed", (1e-3, 4), (1e-3, 4), 
             xlabel=r'log($D_{T}$)', ylabel=r'$D_{P}$', xscale='log', yscale='log')

plot_scatter(true_time, pred_time,
             "y_pred_plots/all_data/Total_Time_Scatter.png",
             "Temporal Vacuity: Predicted vs. Observed", (0, 1), (0, 1), 
             xlabel=r'$T_{T}$', ylabel=r'$T_{P}$')

plot_scatter(true_time, pred_time,
             "y_pred_plots/all_data/Total_Time_Scatter_loglog.png",
             "Temporal Vacuity: Predicted vs. Observed", (1e-3, 1), (1e-3, 1), 
             xlabel=r'$T_{T}$', ylabel=r'$T_{P}$', xscale='log', yscale='log')

plot_scatter(true_time, pred_time,
             "y_pred_plots/all_data/Total_Time_Scatter_Zoomed.png",
             "Temporal Vacuity: Predicted vs. Observed (Zoomed)", (0, 0.2), (0, 0.2), 
             xlabel=r'$T_{T}$', ylabel=r'$T_{P}$')

plot_scatter(true_time, pred_time,
             "y_pred_plots/all_data/Total_Time_Scatter_Zoomed_loglog.png",
             "Temporal Vacuity: Predicted vs. Observed (Zoomed log-log)", 
             (1e-3, 0.2), (1e-3, 0.2), xlabel=r'$T_{T}$', ylabel=r'$T_{P}$', 
             xscale='log', yscale='log')


# Model 2: Tagged features

with open('saved_models/all_data_tagged_model.pkl', 'rb') as f:
    saved_model = pickle.load(f)
    model = saved_model['model']
    x_scaler = saved_model['x_scaler']
    y_scaler = saved_model['y_scaler']

# Prepare data
data = data_sampled.copy()
x = data[['Layer_Thickness', 'Layer_Time', 'Lobe', 'Channel',
          'Wet_Floodplain', 'Dry_Floodplain', 'Marine', 'High_Erosion']].copy()
y = data[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']].copy()

# Normalize inputs
x.loc[:, 'Layer_Thickness'] /= 6.5
x.loc[:, 'Layer_Time'] /= 115

# Predict
x_scaled_values = x_scaler.transform(x)
y_pred_scaled = model.predict(x_scaled_values)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_pred = pd.DataFrame(y_pred, columns=y.columns)

# Calculate vacuity metrics
true_dep = y['Total_Dep'].values / 6.5 - x['Layer_Thickness'].values
pred_dep = y_pred['Total_Dep'].values - x['Layer_Thickness'].values
true_time = y['Total_Time'].values / 115 - x['Layer_Time'].values
pred_time = y_pred['Total_Time'].values - x['Layer_Time'].values

# Generate plots
plot_scatter(true_dep, pred_dep,
             "y_pred_plots/all_tagged_data/Total_Dep_Scatter.png",
             "Depositional Vacuity: Predicted vs. Observed", (0, 3), (0, 3), 
             xlabel=r'$D_{T}$', ylabel=r'$D_{P}$')

plot_scatter(true_dep, pred_dep,
             "y_pred_plots/all_tagged_data/Total_Dep_Scatter_loglog.png",
             "Depositional Vacuity: Predicted vs. Observed", (1e-3, 4), (1e-3, 4), 
             xlabel=r'$D_{T}$', ylabel=r'$D_{P}$', xscale='log', yscale='log')

plot_scatter(true_time, pred_time,
             "y_pred_plots/all_tagged_data/Total_Time_Scatter.png",
             "Temporal Vacuity: Predicted vs. Observed", (0, 1), (0, 1), 
             xlabel=r'$T_{T}$', ylabel=r'$T_{P}$')

plot_scatter(true_time, pred_time,
             "y_pred_plots/all_tagged_data/Total_Time_Scatter_loglog.png",
             "Temporal Vacuity: Predicted vs. Observed", (1e-3, 1), (1e-3, 1), 
             xlabel=r'$T_{T}$', ylabel=r'$T_{P}$', xscale='log', yscale='log')

plot_scatter(true_time, pred_time,
             "y_pred_plots/all_tagged_data/Total_Time_Scatter_Zoomed.png",
             "Temporal Vacuity: Predicted vs. Observed (Zoomed)", (0, 0.2), (0, 0.2), 
             xlabel=r'$T_{T}$', ylabel=r'$T_{P}$')

plot_scatter(true_time, pred_time,
             "y_pred_plots/all_tagged_data/Total_Time_Scatter_Zoomed_loglog.png",
             "Temporal Vacuity: Predicted vs. Observed (Zoomed log-log)", 
             (1e-3, 0.2), (1e-3, 0.2), xlabel=r'$T_{T}$', ylabel=r'$T_{P}$', 
             xscale='log', yscale='log')