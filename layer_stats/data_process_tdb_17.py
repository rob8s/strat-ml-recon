import pandas as pd 
import numpy as np
import time
import scipy.io
from functions.data_split import current_strat
from functions.layer_stat import layer_info
from functions.flatten import flatten
from functions.tag import tag
from functions.error_check import error_check


def load_mat_data(file, key):
    """Load MATLAB file and convert to NumPy array."""
    mat = scipy.io.loadmat(file)
    return np.array(mat[key])


# Load cube and flatten
data = load_mat_data("used_Data/ZdataD_final.mat", 'ZD')
data = flatten(data)

# Add error rate smoothing
data = error_check(data)
data.to_csv("error_rate_data/error_thresh_data.csv", index=False)

# Pull stats from env and data
env = load_mat_data("used_data/Depo_Env.mat", 'DE')
env = flatten(env)
data = pd.read_csv("error_rate_data/error_thresh_data.csv")
data.columns = data.columns.astype(int)
strat, times, start_times, env = current_strat(data, env)

# Save each variable to an individual CSV file
strat.to_csv('error_rate_data/strat.csv', index=False)
times.to_csv('error_rate_data/time.csv', index=False)
start_times.to_csv('error_rate_data/start_times.csv', index=False)
env.to_csv('error_rate_data/env.csv', index=False)

# Load the CSV files back into DataFrames
env = pd.read_csv('used_data/Dep_Env_zeroed.csv')
data = pd.read_csv('used_data/flattened_data_zeroed.csv')
strat = pd.read_csv('used_data/strat_zeroed.csv')
times = pd.read_csv('used_data/times_zeroed.csv')
start_times = pd.read_csv('used_data/start_times_zeroed.csv')

# Calculate layer statistics
start = time.time()
df = layer_info(strat, times, start_times, data, env)
end = time.time()
print(f"Layer stats computation time: {end-start:.2f}s")
df.to_csv("error_rate_data/Layer_Stats_Env.csv", sep=',')

# Remove rows with zero layer thickness
csv_file_path = "error_rate_data/Layer_Stats_Env.csv"
df = pd.read_csv(csv_file_path)
df = df[df['Layer_Thickness'] != 0]
df.to_csv("error_rate_data/Layer_Stats_Env_zeroed.csv", sep=',')

# Apply the tag function to add the new features
df = pd.read_csv("final_data/Layer_Stats_Env_Tagged.csv")
df_tagged = tag(df)
output_csv_path = 'final_data/Layer_Stats_Env_Tagged.csv'
df_tagged.to_csv(output_csv_path, index=False)
