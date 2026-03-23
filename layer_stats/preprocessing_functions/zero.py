import pandas as pd

# Load the data from error_check_data folder
data = pd.read_csv('error_check_data/error_thresh_data.csv')
print("Data shape:", data.shape)
strat = pd.read_csv('error_check_data/strat.csv')
print("Strat shape:", strat.shape)
times = pd.read_csv('error_check_data/time.csv')
print("Times shape:", times.shape)
start_times = pd.read_csv('error_check_data/start_times.csv')
print("Start Times shape:", start_times.shape)


# Find all rows with non-zero values in strat and filter them from all dataframes
non_zero_indices = strat.index[strat.sum(axis=1) != 0]

strat_zeroed = strat.loc[non_zero_indices]
times_zeroed = times.loc[non_zero_indices]
start_times_zeroed = start_times.loc[non_zero_indices]
data_zeroed = data.loc[non_zero_indices]

# Print shapes of the new dataframes
print("Strat Zeroed shape:", strat_zeroed.shape)
print("Times Zeroed shape:", times_zeroed.shape)
print("Start Times Zeroed shape:", start_times_zeroed.shape)
print("Lobes Zeroed shape:", lobes_zeroed.shape)
print("Channels Zeroed shape:", channels_zeroed.shape)
print("Data Zeroed shape:", data_zeroed.shape)

# Save the new dataframes with a _zeroed suffix
strat_zeroed.to_csv('error_check_data/strat_zeroed.csv', index=False)
times_zeroed.to_csv('error_check_data/times_zeroed.csv', index=False)
start_times_zeroed.to_csv('error_check_data/start_times_zeroed.csv', index=False)
lobes_zeroed.to_csv('error_check_data/lobes_zeroed.csv', index=False)
channels_zeroed.to_csv('error_check_data/channels_zeroed.csv', index=False)
data_zeroed.to_csv('error_check_data/flattened_data_zeroed.csv', index=False)

print("Data saved to CSV files successfully!")
