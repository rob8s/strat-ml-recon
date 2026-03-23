import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load environment-split datasets
datasets = {
    'lobe': pd.read_csv('data/final_data/env_split_data/lobe_tagged.csv'),
    'channel': pd.read_csv('data/final_data/env_split_data/channel_tagged.csv'),
    'wet_floodplain': pd.read_csv('data/final_data/env_split_data/wet_floodplain_tagged.csv'),
    'dry_floodplain': pd.read_csv('data/final_data/env_split_data/dry_floodplain_tagged.csv'),
    'marine': pd.read_csv('data/final_data/env_split_data/marine_tagged.csv')
}

# Normalization factors
thickness_scaling_factor = 6.5
time_scaling_factor = 115.0

# Model features and targets
features = ['Layer_Thickness', 'Layer_Time', 'High_Erosion']
targets = ['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']

# Hyperparameters for non-marine environments
best_params_common = {
    'n_estimators': 413,
    'max_depth': 16,
    'min_samples_split': 13,
    'min_samples_leaf': 1,
    'max_features': 'sqrt'
}

# Hyperparameters for marine environment
best_params_marine = {
    'n_estimators': 289,
    'max_depth': 19,
    'min_samples_split': 16,
    'min_samples_leaf': 16,
    'max_features': None
}

# Train and evaluate models for each environment
for name, data in datasets.items():
    print(f"\nTraining Random Forest on {name} dataset")

    # Convert data types
    data = data.astype({
        'Layer_Thickness': float,
        'Layer_Time': float,
        'Total_Dep': float,
        'Total_Time': float,
        'Stasis_Proportion': float,
        'Deposition_Proportion': float,
        'High_Erosion': int,
        'High_Stasis': int
    })

    # Normalize features and targets
    data.loc[:, 'Layer_Thickness'] /= thickness_scaling_factor
    data.loc[:, 'Total_Dep'] /= thickness_scaling_factor
    data.loc[:, 'Layer_Time'] /= time_scaling_factor
    data.loc[:, 'Total_Time'] /= time_scaling_factor

    # Extract features and targets
    X = data[features]
    y = data[targets]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select hyperparameters based on environment
    best_params = best_params_marine if name == 'marine' else best_params_common

    # Train model
    rf = RandomForestRegressor(random_state=42, **best_params)
    multi_rf = MultiOutputRegressor(rf)
    multi_rf.fit(X_train, y_train)

    # Save model
    model_filename = f'saved_models/{name}_env_split_tagged_model.pkl'
    joblib.dump(multi_rf, model_filename)

    # Predict and evaluate
    y_pred = multi_rf.predict(X_test)
    
    # Inverse scaling
    y_pred[:, 0] *= thickness_scaling_factor
    y_test['Total_Dep'] *= thickness_scaling_factor
    y_pred[:, 1] *= time_scaling_factor
    y_test['Total_Time'] *= time_scaling_factor

    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')

    print("Mean Absolute Error for each target:")
    for target, error in zip(targets, mae):
        print(f'  {target}: {error:.4f}')

    print("R-squared score for each target:")
    for target, score in zip(targets, r2):
        print(f'  {target}: {score:.4f}')

print("\nTraining completed.")
