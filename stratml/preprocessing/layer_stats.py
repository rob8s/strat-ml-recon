import pandas as pd 
import numpy as np
import time
import scipy.io

#Takes Layer Thickness, time to form, and env element
#Returns total dep, total time, and proportion of each layer is stasis
def layer_info(strat, time, start, data, env):
    # Data dimensions
    rows, cols = strat.shape
    data_row, data_col = data.shape
    
    # Layer thickness, time to form, env type
    layers = pd.DataFrame(0, index=range(rows*cols), columns=['Layer_Thickness', 'Layer_Time', 'Marine', 'Dry_Floodplain', 'Wet_Floodplain', 'Lobe', 'Channel', 'Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], dtype=float)

    # Starting position
    pos = 0
    for index in range(rows):
        for col in range(cols):
            # Skip if strat value is 0
            if strat.iat[index, col] == 0:
                continue
            else:
                # Targets
                layers.at[pos, 'Layer_Thickness'] = float(strat.iat[index, col])
                layers.at[pos, 'Layer_Time'] = float(time.iat[index, col])

                # Indexing patch
                if (col + 1 >= cols) or (start.iat[index, col + 1] == 0):
                    end = data_col - 1
                else:
                    end = start.iat[index, col + 1]

                # First position for each layer events
                data_pos = int(start.iat[index, col])
                stasis_count = 0 
                dep_count = 0
                count = 0
                env_avg = []

                # For all events in a given time frame
                for x in range(int(end - start.iat[index, col])):
                    # Increase total deposition
                    if data.iat[index, data_pos] > data.iat[index, data_pos - 1]:
                        layers.at[pos, 'Total_Dep'] += data.iat[index, data_pos] - data.iat[index, data_pos - 1]
                        dep_count += 1

                    # Add more time of stasis
                    if data.iat[index, data_pos] == data.iat[index, data_pos - 1]:
                        stasis_count += 1

                    # Total time increase and next event
                    count += 1
                    data_pos += 1
                    env_avg.append(env.iat[index, col])

                # Divide by 0 fix
                if count == 0:
                    count = 1
                else:
                    env_most = env_avg[np.argmax(env_avg)]
                    if env_most == 0:
                        layers.at[pos, 'Marine'] = 1
                    if env_most == 1:
                        layers.at[pos, 'Dry_Floodplain'] = 1
                    if env_most == 2: 
                        layers.at[pos, 'Wet_Floodplain'] = 1
                    if env_most == 3:
                        layers.at[pos, 'Lobe'] = 1
                    if env_most == 4:
                        layers.at[pos, 'Channel'] = 1

                # Add total time behind each layer and proportion of stasis to total time
                layers.at[pos, 'Total_Time'] = count
                layers.at[pos, 'Stasis_Proportion'] = stasis_count / count
                layers.at[pos, 'Deposition_Proportion'] = dep_count / count
                # Next line
                pos += 1

    return layers