import pandas as pd 
import numpy as np
import time
import scipy.io

#Takes in current Strat, time to form strat, and start time for each layer
#returns each layer with the events to form that layer
def layer_conversion(strat, time, start, data, lobes, channels):
    # Data dimensions
    rows, cols = strat.shape
    data_row, data_col = data.shape
    
    # Layer thickness, time to form, env type
    layer_info = pd.DataFrame(0, index=range(rows*cols), columns=['Layer_Thickness', 'Layer_Time', 'Lobe', 'Channel'], dtype=float)
    
    # Erosion, deposition, stasis
    event_info = pd.DataFrame(np.nan, index=range(rows*cols), columns=np.arange(50), dtype=float)
    
    # Starting position
    pos = 0
    count = 0
    chunk_size = 1000  # Save data in chunks
    
    # Write the CSV header once at the beginning
    with open("final_data/Layer_Events.csv", "w") as f:
        combined_columns = list(layer_info.columns) + list(event_info.columns)
        pd.DataFrame(columns=combined_columns).to_csv(f, sep=',', index=False)
    
    for index in range(rows):
        for col in range(cols):
            # Skip rows where strat is 0
            if strat.iat[index, col] == 0:
                continue

            # Targets
            layer_info.iat[pos, 0] = strat.iat[index, col]  # Layer_Thickness
            layer_info.iat[pos, 1] = time.iat[index, col]   # Layer_Time
            layer_info.iat[pos, 2] = lobes.iat[index, col]  # Lobe (no conversion)
            layer_info.iat[pos, 3] = channels.iat[index, col]  # Channel (no conversion)
            
            # Indexing patch
            if (col+1 >= cols) or (start.iat[index, col+1] == 0):
                end = data_col-1
            else:
                end = start.iat[index, col+1]
            
            # First position for each layer events
            data_pos = int(start.iat[index, col])
            event = 0
            
            # For all events in a given time frame
            while data_pos < end and event < event_info.shape[1]:
                event_info.iat[pos, event] = data.iat[index, data_pos] - data.iat[index, data_pos-1]
                data_pos += 1
                event += 1
                
            pos += 1
        
        count += 1
        
        if count % chunk_size == 0:
            # Drop all columns of 0's
            non_zero_columns = event_info.loc[:, (event_info != 0).any(axis=0)].columns
            event_info = event_info[non_zero_columns]
            
            # Drop all rows of all 0's and NA
            layer_info = layer_info.loc[(layer_info != 0).any(axis=1)]
            event_info = event_info.dropna(how='all')
            
            combined_data = pd.concat([layer_info, event_info], axis=1)
            combined_data.to_csv("final_data/Layer_Events.csv", sep=',', index=False, mode='a', header=False)
            print(f"Processed {count} entries")
            
            # Reset for next chunk
            layer_info = pd.DataFrame(0, index=range(rows*cols), columns=['Layer_Thickness', 'Layer_Time', 'Lobe', 'Channel'], dtype=float)
            event_info = pd.DataFrame(np.nan, index=range(rows*cols), columns=np.arange(50), dtype=float)
            pos = 0
    
    # Final save for remaining data
    if pos > 0:
        # Drop all columns of 0's
        non_zero_columns = event_info.loc[:, (event_info != 0).any(axis=0)].columns
        event_info = event_info[non_zero_columns]
        
        # Drop all rows of all 0's and NA
        layer_info = layer_info.loc[(layer_info != 0).any(axis=1)]
        event_info = event_info.dropna(how='all')
        
        combined_data = pd.concat([layer_info, event_info], axis=1)
        combined_data.to_csv("final_data/Layer_Events.csv", sep=',', index=False, mode='a', header=False)
        print(f"Processed {count} entries")

    return combined_data