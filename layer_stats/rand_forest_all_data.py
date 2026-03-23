import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
import os
import pickle

# Load and preprocess data
csv_file_path = 'data/final_data/Layer_Stats_Env_Tagged.csv'
dfg = pd.read_csv(csv_file_path)

# Filter data: exclude marine deposits and small layers
dfg = dfg[dfg['Marine'] != 1]
dfg = dfg[dfg['Layer_Thickness'] >= 0.065]

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
thickness_scaling_factor = 6.5
time_scaling_factor = 115.0

dfg['Layer_Thickness'] /= thickness_scaling_factor
dfg['Total_Dep'] /= thickness_scaling_factor
dfg['Layer_Time'] /= time_scaling_factor
dfg['Total_Time'] /= time_scaling_factor

# Sample data
dfg_sampled = dfg.sample(n=100_000, random_state=42)

# Extract features and targets
x = dfg_sampled[['Layer_Thickness', 'Layer_Time', 'Lobe', 'Channel', 'Wet_Floodplain', 'Dry_Floodplain', 'Marine']]
y = dfg_sampled[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]

# Scale features and targets
x_scaler = RobustScaler()
y_scaler = RobustScaler()
x_scaled = x_scaler.fit_transform(x)
y_scaled = y_scaler.fit_transform(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(
    n_estimators=154,
    max_depth=18,
    min_samples_split=13,
    min_samples_leaf=19,
    max_features=None,
    random_state=42
)
multi_rf = MultiOutputRegressor(rf)
multi_rf.fit(x_train, y_train)

# Predict and evaluate
y_pred_scaled = multi_rf.predict(x_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test_unscaled = y_scaler.inverse_transform(y_test)

# Evaluate model
mae = mean_absolute_error(y_test_unscaled, y_pred, multioutput='raw_values')
r2 = r2_score(y_test_unscaled, y_pred, multioutput='raw_values')

target_variables = ['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']
print("Mean Absolute Error for each target:")
for target, error in zip(target_variables, mae):
    print(f'{target}: {error:.4f}')

print("\nR-squared score for each target:")
for target, score in zip(target_variables, r2):
    print(f'{target}: {score:.4f}')

# Save model
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)
model_file_path = os.path.join(model_dir, 'all_data_model.pkl')

with open(model_file_path, 'wb') as f:
    pickle.dump({
        'model': multi_rf,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler
    }, f)