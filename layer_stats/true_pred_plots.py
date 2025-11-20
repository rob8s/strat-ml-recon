import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and scalers
with open('saved_models/all_data_model.pkl', 'rb') as f:
    saved_model = pickle.load(f)
    model = saved_model['model']
    x_scaler = saved_model['x_scaler']
    y_scaler = saved_model['y_scaler']

# Load the data
csv_file_path = 'data/final_data/Layer_Stats_Env_Tagged.csv'
data = pd.read_csv(csv_file_path)

# Sample data
data = data.sample(n=1_000, random_state=42).copy()

# Select input and target features
x = data[['Layer_Thickness', 'Layer_Time', 'Lobe', 'Channel', 'Wet_Floodplain', 'Dry_Floodplain', 'Marine']].copy()
y = data[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']].copy()

# Normalize input features
x_scaled = x.copy()
x_scaled.loc[:, 'Layer_Thickness'] /= 6.5
x_scaled.loc[:, 'Layer_Time'] /= 115

# Scale and predict
x_scaled_values = x_scaler.transform(x_scaled)
y_pred_scaled = model.predict(x_scaled_values)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_pred = pd.DataFrame(y_pred, columns=y.columns)

# Compute differences in original units
true_values_dep = y['Total_Dep'].values / 6.5 - x_scaled['Layer_Thickness'].values
pred_values_dep = y_pred['Total_Dep'].values - x_scaled['Layer_Thickness'].values

true_values_time = y['Total_Time'].values / 115 - x_scaled['Layer_Time'].values
pred_values_time = y_pred['Total_Time'].values - x_scaled['Layer_Time'].values

# Color Palette
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# Plot for Total_Dep
plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(true_values_dep, pred_values_dep, alpha=0.4, edgecolor='k', s=7, label="Predictions")
plt.plot([0, 1.5], [0, 1.5], linestyle="--", color="red", linewidth=1, label="Perfect Prediction (1:1)")
plt.xlabel('True Values', fontsize=16)
plt.ylabel('Predicted Values', fontsize=16)
plt.title("Missing Deposition Prediction", fontsize=16)
plt.xlim(0, 1.5)
plt.ylim(0, 1.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tick_params(direction='in', length=5, width=1)
plt.grid(True, linewidth=1, alpha=1)
plt.legend(fontsize=12, loc="upper left")
plt.savefig("y_pred_plots/all_data/Total_Dep_Scatter.png", dpi=300, bbox_inches="tight")

# Plot for Total_Time
plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(true_values_time, pred_values_time, alpha=0.4, edgecolor='k', s=7, label="Predictions")
plt.plot([0, 1], [0, 1], linestyle="--", color="red", linewidth=1, label="Perfect Prediction (1:1)")
plt.xlabel('True Values', fontsize=16)
plt.ylabel('Predicted Values', fontsize=16)
plt.title("Missing Time Prediction", fontsize=16)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tick_params(direction='in', length=5, width=1)
plt.grid(True, linewidth=1, alpha=1)
plt.legend(fontsize=12, loc="upper left")
plt.savefig("y_pred_plots/all_data/Total_Time_Scatter.png", dpi=300, bbox_inches="tight")

# Zoomed plot for Total_Time
plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(true_values_time, pred_values_time, alpha=0.4, edgecolor='k', s=7, label="Predictions")
plt.plot([0, 0.2], [0, 0.2], linestyle="--", color="red", linewidth=1, label="Perfect Prediction (1:1)")
plt.xlabel('True Values', fontsize=16)
plt.ylabel('Predicted Values', fontsize=16)
plt.title("Missing Time Prediction (zoomed)", fontsize=16)
plt.xlim(0, 0.2)
plt.ylim(0, 0.2)
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tick_params(direction='in', length=5, width=1)
plt.grid(True, linewidth=1, alpha=1)
plt.legend(fontsize=12, loc="upper left")
plt.savefig("y_pred_plots/all_data/Total_Time_Scatter_Zoomed.png", dpi=300, bbox_inches="tight")


##################################################################################################

# Load the model and scalers
with open('saved_models/all_data_tagged_model.pkl', 'rb') as f:
    saved_model = pickle.load(f)
    model = saved_model['model']
    x_scaler = saved_model['x_scaler']
    y_scaler = saved_model['y_scaler']

# Load the data
csv_file_path = 'data/final_data/Layer_Stats_Env_Tagged.csv'
data = pd.read_csv(csv_file_path)
# Sample data
data = data.sample(n=1_000, random_state=42)
x = data[['Layer_Thickness', 'Layer_Time', 'Lobe', 'Channel', 'Wet_Floodplain', 'Dry_Floodplain', 'Marine', 'High_Erosion']]
y = data[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x.loc[:, 'Layer_Thickness'] /= 6.5
x.loc[:, 'Layer_Time'] /= 115

x_scaled = x_scaler.transform(x)
y_pred_scaled = model.predict(x_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_pred = pd.DataFrame(y_pred, columns=y.columns)


# Compute differences in original units
true_values_dep = y['Total_Dep'].values / 6.5 - x['Layer_Thickness'].values
pred_values_dep = y_pred['Total_Dep'].values - x['Layer_Thickness'].values

true_values_time = y['Total_Time'].values / 115 - x['Layer_Time'].values
pred_values_time = y_pred['Total_Time'].values - x['Layer_Time'].values

# Plot for Total_Dep (Second Model)
plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(true_values_dep, pred_values_dep, alpha=0.4, edgecolor='k', s=7, label="Predictions")
plt.plot([0, 1.5], [0, 1.5], linestyle="--", color="red", linewidth=1, label="Perfect Prediction (1:1)")
plt.xlabel('True Values', fontsize=16, labelpad=12)
plt.ylabel('Predicted Values', fontsize=16, labelpad=12)
plt.title("Missing Deposition Prediction", fontsize=16, pad=15)
plt.xlim(0, 1.5)
plt.ylim(0, 1.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tick_params(direction='in', length=5, width=1)
plt.grid(True, linewidth=1, alpha=1)
plt.legend(fontsize=12, loc="upper left")
plt.savefig("y_pred_plots/all_tagged_data/Total_Dep_Scatter.png", dpi=300, bbox_inches="tight")

# Plot for Total_Time (Full Range, Second Model)
plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(true_values_time, pred_values_time, alpha=0.4, edgecolor='k', s=7, label="Predictions")
plt.plot([0, 1], [0, 1], linestyle="--", color="red", linewidth=1, label="Perfect Prediction (1:1)")
plt.xlabel('True Values', fontsize=16, labelpad=12)
plt.ylabel('Predicted Values', fontsize=16, labelpad=12)
plt.title("Missing Time Prediction", fontsize=16, pad=15)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tick_params(direction='in', length=5, width=1)
plt.grid(True, linewidth=1, alpha=1)
plt.legend(fontsize=12, loc="upper left")
plt.savefig("y_pred_plots/all_tagged_data/Total_Time_Scatter.png", dpi=300, bbox_inches="tight")

# Zoomed Plot for Total_Time (Second Model)
plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(true_values_time, pred_values_time, alpha=0.4, edgecolor='k', s=7, label="Predictions")
plt.plot([0, 0.2], [0, 0.2], linestyle="--", color="red", linewidth=1, label="Perfect Prediction (1:1)")
plt.xlabel('True Values', fontsize=16, labelpad=12)
plt.ylabel('Predicted Values', fontsize=16, labelpad=12)
plt.title("Missing Time Prediction (Zoomed)", fontsize=16, pad=15)
plt.xlim(0, 0.2)
plt.ylim(0, 0.2)
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tick_params(direction='in', length=5, width=1)
plt.grid(True, linewidth=1, alpha=1)
plt.legend(fontsize=12, loc="upper left")
plt.savefig("y_pred_plots/all_tagged_data/Total_Time_Scatter_Zoomed.png", dpi=300, bbox_inches="tight")


