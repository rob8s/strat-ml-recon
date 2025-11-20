import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

def load_model(file_path):
    #load model
    model = joblib.load(file_path)
    return model

#non-dimensionalize
def scale(x):
    x['Layer_Thickness'] /= 6.5
    x['Layer_Time'] /= 115
    return x

#redimensionalize
def rescale(y_pred):
    y_pred[:, 0] *= 6.5
    y_pred[:, 1] *= 115
    return y_pred

# Load data
csv_file_path = 'data/final_data/Layer_Stats_Env_Tagged.csv'
data = pd.read_csv(csv_file_path)

#########################################################################################################################
# All Data
model = load_model('saved_models/all_data_model.pkl')

#Sample
data_all = data.sample(100_000, random_state=42)

# Split data and non-dimensionalize
x = data_all[['Layer_Thickness', 'Layer_Time', 'Lobe', 'Channel', 'Wet_Floodplain', 'Dry_Floodplain', 'Marine']]
y = data_all[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale targets
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Save the results to a text file
with open("performance_metrics.txt", "w") as f:
    f.write("All Data:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")
    f.write("\n##########################################################\n")
#######################################################################################################################
# All Data Tagged
model = load_model('saved_models/all_data_tagged_model.pkl')

#Sample
data_tagged = data.sample(100_000, random_state=84)

# Split data and non-dimensionalize
x = data_tagged[['Layer_Thickness', 'Layer_Time', 'Lobe', 'Channel', 'Wet_Floodplain', 'Dry_Floodplain', 'Marine', 'High_Erosion']]
y = data_tagged[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Append the results to the same text file
with open("performance_metrics.txt", "a") as f:
    f.write("\nAll Data Tagged:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")
    f.write("\n##########################################################\n")

#######################################################################################################################
csv_file_path = 'data/final_data/env_split_data/channel_tagged.csv'
data = pd.read_csv(csv_file_path)
#######################################################################################################################
# Channel
model = load_model('saved_models/channel_env_split_model.pkl')

#Sample
data_tagged = data.sample(100_000, random_state=84)

# Split data and non-dimensionalize
x = data_tagged[['Layer_Thickness', 'Layer_Time']]
y = data_tagged[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Append the results to the same text file
with open("performance_metrics.txt", "a") as f:
    f.write("\nChannel Env:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")

#######################################################################################################################
# Channel Tagged
model = load_model('saved_models/channel_env_split_tagged_model.pkl')

#Sample
data_tagged = data.sample(100_000, random_state=84)

# Split data and non-dimensionalize
x = data_tagged[['Layer_Thickness', 'Layer_Time', 'High_Erosion']]
y = data_tagged[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Append the results to the same text file
with open("performance_metrics.txt", "a") as f:
    f.write("\nChannel Env Tagged:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")
    f.write("\n##########################################################\n")
#######################################################################################################################
csv_file_path = 'data/final_data/env_split_data/dry_floodplain_tagged.csv'
data = pd.read_csv(csv_file_path)
#######################################################################################################################
#dry floodplain
model = load_model('saved_models/dry_floodplain_env_split_model.pkl')

#Sample
data_tagged = data.sample(100_000, random_state=84)

# Split data and non-dimensionalize
x = data_tagged[['Layer_Thickness', 'Layer_Time']]
y = data_tagged[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Append the results to the same text file
with open("performance_metrics.txt", "a") as f:
    f.write("\nDry Floodplain Env:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")

#######################################################################################################################
# dry floodplain tagged 
model = load_model('saved_models/dry_floodplain_env_split_tagged_model.pkl')

#Sample
data_tagged = data.sample(100_000, random_state=84)

# Split data and non-dimensionalize
x = data_tagged[['Layer_Thickness', 'Layer_Time', 'High_Erosion']]
y = data_tagged[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Append the results to the same text file
with open("performance_metrics.txt", "a") as f:
    f.write("\nDry Floodplain Env Tagged:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")
    f.write("\n##########################################################\n")

#######################################################################################################################
csv_file_path = 'data/final_data/env_split_data/wet_floodplain_tagged.csv'
data = pd.read_csv(csv_file_path)
#######################################################################################################################
#wet floodplain
model = load_model('saved_models/wet_floodplain_env_split_model.pkl')

#Sample
data_tagged = data.sample(100_000, random_state=84)

# Split data and non-dimensionalize
x = data_tagged[['Layer_Thickness', 'Layer_Time']]
y = data_tagged[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Append the results to the same text file
with open("performance_metrics.txt", "a") as f:
    f.write("\nWet Floodplain Env:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")

#######################################################################################################################
# wet floodplain tagged 
model = load_model('saved_models/wet_floodplain_env_split_tagged_model.pkl')

#Sample
data_tagged = data.sample(100_000, random_state=84)

# Split data and non-dimensionalize
x = data_tagged[['Layer_Thickness', 'Layer_Time', 'High_Erosion']]
y = data_tagged[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Append the results to the same text file
with open("performance_metrics.txt", "a") as f:
    f.write("\nWet Floodplain Env Tagged:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")
    f.write("\n##########################################################\n")

#######################################################################################################################
csv_file_path = 'data/final_data/env_split_data/lobe_tagged.csv'
data = pd.read_csv(csv_file_path)
#######################################################################################################################
#lobe
model = load_model('saved_models/lobe_env_split_model.pkl')

#Sample
data_tagged = data.sample(100_000, random_state=84)

# Split data and non-dimensionalize
x = data_tagged[['Layer_Thickness', 'Layer_Time']]
y = data_tagged[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Append the results to the same text file
with open("performance_metrics.txt", "a") as f:
    f.write("\nLobe Env:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")

#######################################################################################################################
#lobe tagged
model = load_model('saved_models/lobe_env_split_tagged_model.pkl')

#Sample
data_tagged = data.sample(100_000, random_state=84)

# Split data and non-dimensionalize
x = data_tagged[['Layer_Thickness', 'Layer_Time', 'High_Erosion']]
y = data_tagged[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Append the results to the same text file
with open("performance_metrics.txt", "a") as f:
    f.write("\nLobe Env Tagged:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")
    f.write("\n##########################################################\n")

#######################################################################################################################
csv_file_path = 'data/final_data/env_split_data/marine_tagged.csv'
data = pd.read_csv(csv_file_path)
#######################################################################################################################
#marine
model = load_model('saved_models/marine_env_split_model.pkl')

#Sample
data_tagged = data.sample(100_000, random_state=84)

# Split data and non-dimensionalize
x = data_tagged[['Layer_Thickness', 'Layer_Time']]
y = data_tagged[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Append the results to the same text file
with open("performance_metrics.txt", "a") as f:
    f.write("\nMarine Env:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")

#######################################################################################################################
#marine tagged
model = load_model('saved_models/marine_env_split_tagged_model.pkl')

#Sample
data_tagged = data.sample(100_000, random_state=84)

# Split data and non-dimensionalize
x = data_tagged[['Layer_Thickness', 'Layer_Time', 'High_Erosion']]
y = data_tagged[['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion']]
x = scale(x)

# Predict, rescale, convert output to df for easy of plotting
y_pred = model.predict(x)
#rescale
y_pred = rescale(y_pred)
y_pred = pd.DataFrame(y_pred, columns=['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'])

# Calculate R2 score and MAE for each target
r2 = r2_score(y, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y, y_pred, multioutput='raw_values')

# Append the results to the same text file
with open("performance_metrics.txt", "a") as f:
    f.write("\nMarine Env Tagged:\n")
    f.write("\nMean Absolute Error (MAE):\n")
    for target, error in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], mae):
        f.write(f"{target}: {error:.4f}\n")

    f.write("\nR² Scores:\n")
    for target, score in zip(['Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], r2):
        f.write(f"{target}: {score:.4f}\n")
    f.write("\n##########################################################\n")