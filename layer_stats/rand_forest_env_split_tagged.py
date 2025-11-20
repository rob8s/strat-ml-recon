import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib  # Import joblib for saving models

# Load datasets
datasets = {
    'lobe': pd.read_csv('final_data/env_split_data/lobe_tagged.csv'),
    'channel': pd.read_csv('final_data/env_split_data/channel_tagged.csv'),
    'wet_floodplain': pd.read_csv('final_data/env_split_data/wet_floodplain_tagged.csv'),
    'dry_floodplain': pd.read_csv('final_data/env_split_data/dry_floodplain_tagged.csv'),
    'marine': pd.read_csv('final_data/env_split_data/marine_tagged.csv')
}

# Scaling factors
thickness_scaling_factor = 6.5
time_scaling_factor = 115.0

# Features and target columns
features = ['Layer_Thickness', 'Layer_Time', 'High_Erosion']
targets = ['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']

# Best parameters for all datasets except 'marine'
best_params_common = {
    'n_estimators': 413,
    'max_depth': 16,
    'min_samples_split': 13,
    'min_samples_leaf': 1,
    'max_features': 'sqrt'
}

# Best parameters for the 'marine' dataset
best_params_marine = {
    'n_estimators': 289,
    'max_depth': 19,
    'min_samples_split': 16,
    'min_samples_leaf': 16,
    'max_features': None
}

# Iterate through each dataset
for name, data in datasets.items():
    print(f"\nRunning Random Forest on dataset: {name}")

    # Data type conversion
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

    # Apply scaling using .loc to avoid SettingWithCopyWarning
    data.loc[:, 'Layer_Thickness'] /= thickness_scaling_factor
    data.loc[:, 'Total_Dep'] /= thickness_scaling_factor
    data.loc[:, 'Layer_Time'] /= time_scaling_factor
    data.loc[:, 'Total_Time'] /= time_scaling_factor

    # Define features and targets
    X = data[features]
    y = data[targets]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use the appropriate parameters for 'marine' and the common parameters for others
    if name == 'marine':
        best_params = best_params_marine
    else:
        best_params = best_params_common

    # Define and train the RandomForestRegressor with the best parameters
    rf = RandomForestRegressor(random_state=42, **best_params)
    multi_rf_best = MultiOutputRegressor(rf)
    multi_rf_best.fit(X_train, y_train)

    # Print the type of model before saving
    print(f"Model type before saving: {type(multi_rf_best)}")  # Check the model type

    # Save the trained model to a pickle file
    model_filename = f'saved_models/{name}_env_split_tagged_model.pkl'  # Specify the path and filename
    joblib.dump(multi_rf_best, model_filename)  # Save the model

    # Predict
    y_pred = multi_rf_best.predict(X_test)

    # Reverse scaling for y_pred and y_test
    y_pred[:, 0] *= thickness_scaling_factor  # Total_Dep
    y_test['Total_Dep'] *= thickness_scaling_factor

    y_pred[:, 1] *= time_scaling_factor       # Total_Time
    y_test['Total_Time'] *= time_scaling_factor

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')

    # Print the results
    print("Mean Absolute Error for each target:")
    for target, error in zip(targets, mae):
        print(f'{target}: {error:.4f}')

    print("R2 Score for each target:")
    for target, score in zip(targets, r2):
        print(f'{target}: {score:.4f}')

print("Random Forest evaluation completed.")
