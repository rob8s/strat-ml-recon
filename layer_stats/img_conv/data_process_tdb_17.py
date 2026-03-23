import pandas as pd 
import numpy as np
import scipy.io

##############################################################################################################################################  

#Takes in depth at each time
#Returns ending stratigraphy, time to form each layer, and start time of each layer
def current_strat(data, lobe_data, channel_data):
    rows, cols = data.shape
    #current strat
    strat = pd.DataFrame(0, index=range(rows), columns=range(cols), dtype = float)
    #layer time to form
    time = pd.DataFrame(0, index=range(rows), columns=range(cols), dtype = float)
    #starting time of each layer
    start_times = pd.DataFrame(0, index=range(rows), columns=range(cols), dtype = float)
    #does layer sit under a lobe
    lobes = pd.DataFrame(0, index=range(rows), columns=range(cols), dtype = int)
    #does layer sit under a chanel
    channels = pd.DataFrame(0, index=range(rows), columns=range(cols), dtype = int)

    #each data point
    for index in range(rows):
        #prev depth for comparison
        prev_change = data.at[index, 0]
        #current layer
        layer_index = 0
        #For start time of each layer
        new_start = True

        #each measurement
        for col in range(cols):
            #start times for each layer
            if (new_start == True) and (data.at[index, col] > prev_change):
                start_times.at[index, layer_index] = col
                
                #check if lobes or channel exist
                if lobe_data.at[index,col] == 1:
                    lobes.at[index, layer_index] = 1
                if channel_data.at[index,col] == 1:
                    channels.at[index, layer_index] = 1
                new_start = False

            #deposition
            if data.at[index, col] > prev_change:
                #update layer thickness, update time taken to form
                strat.at[index, layer_index] += (data.at[index, col] - prev_change)
                time.at[index, layer_index] += 1

                #check if lobes or channel exist
                if lobe_data.at[index,col] == 1:
                    lobes.at[index, layer_index] = 1
                if channel_data.at[index,col] == 1:
                    channels.at[index, layer_index] = 1
                #saved previous value for eval
                prev_change = data.at[index, col]

            #stasis     
            elif data.at[index, col] == prev_change:
                prev_change = data.at[index, col]

            #erosion
            elif data.at[index, col] < prev_change:
                #total erosion
                erosion = prev_change - data.at[index, col]
                #require all erosion to occur
                while erosion != 0:
                    #no overdraw if already at the bottom of the strat
                    if layer_index == 0 & (erosion >= strat.at[index, layer_index]):
                        strat.at[index, layer_index] = 0
                        time.at[index, layer_index] = 0
                        start_times.at[index, layer_index] = 0
                        lobes.at[index, layer_index] = 0
                        channels.at[index, layer_index] = 0
                        erosion = 0

                    #too much erosion for given layer, spill to next layer
                    elif erosion >= strat.at[index, layer_index]:
                        #zero out that layer, and move to next layer
                        erosion -= strat.at[index, layer_index]
                        strat.at[index, layer_index] = 0
                        time.at[index, layer_index] = 0
                        start_times.at[index, layer_index] = 0
                        lobes.at[index, layer_index] = 0
                        channels.at[index, layer_index] = 0
                        layer_index -= 1    
                        
                    #erode of current layer, average time removal too
                    elif erosion < strat.at[index, layer_index]:
                        #subtract from current layer
                        time.at[index, layer_index] = time.at[index, layer_index] - ((erosion / strat.at[index, layer_index])*time.at[index, layer_index])
                        strat.at[index, layer_index] -= erosion
                        erosion = 0
                
                prev_change = data.at[index, col]
                layer_index += 1
                new_start = True

    #drop all columns of 0's
    all_zeros = (strat == 0).all()
    zero_columns = all_zeros[all_zeros].index
    strat.drop(columns=zero_columns, inplace=True)
    time.drop(columns=zero_columns, inplace=True)
    start_times.drop(columns=zero_columns, inplace=True)
    lobes.drop(columns=zero_columns, inplace=True)
    channels.drop(columns=zero_columns, inplace=True)

    #set column names starting at 0, for indexing in next step
    new_column_names = list(range(len(strat.columns)))
    strat.columns = new_column_names
    time.columns = new_column_names
    start_times.columns = new_column_names
    lobes.columns = new_column_names
    channels.columns = new_column_names

    return strat, time, start_times, lobes, channels

###################################################################################################################################################

#Takes in current Strat, time to form strat, and start time for each layer
#returns each layer with the events to form that layer
def layer_conversion(strat, time, start, data, lobes, channels):
    #data dimensions
    rows, cols = strat.shape
    data_row, data_col = data.shape
    #Layer thickness, time to form, env type
    layer_info = pd.DataFrame(0, index=range(rows*cols), columns=['Layer_Thickness', 'Layer_Time'], dtype = float)
    #erosion, deposition, stasis
    event_info = pd.DataFrame(np.nan, index=range(rows*cols), columns=np.arange(50), dtype = float)

    #starting position
    pos = 0
    for index in range(rows):
        for col in range(cols):
            #0 layers are contained in the data at the ends of some rows
            if strat.at[index, col] == 0:
                None
            else:
                #targets
                layer_info.at[pos, 'Layer_Thickness'] = strat.at[index, col]
                layer_info.at[pos, 'Layer_Time'] = time.at[index, col]
                
                #indexing patch
                if (col+1 >= cols) or (start.at[index, col+1] == 0):
                    end = data_col-1
                else:
                    end = start.at[index, col+1]
                
                #first position for each layer events
                data_pos = int(start.at[index, col])
                event = 0
                #for all events in a given time frame
                for x in range(int(end - start.at[index, col])):
                    event_info.at[pos, event] = data.at[index, data_pos] - data.at[index, data_pos-1]
                    data_pos += 1
                    event += 1
                pos += 1


    #drop all columns of 0's
    all_zeros = (event_info == 0).all()
    zero_columns = all_zeros[all_zeros].index
    event_info.drop(columns=zero_columns, inplace=True)

    #drop all rows of all 0's and NA
    layer_info = layer_info.loc[(layer_info != 0).any(axis=1)]
    event_info = event_info.dropna(how='all')
    
    combined_data = pd.concat([layer_info, event_info], axis=1)
    return combined_data
################################################################################################################################

#Takes Layer Thickness, time to form, and env element
#Returns total dep, total time, and proportion of each layer is stasis
def layer_info(strat, time, start, data, lobes, channels):
    #data dimensions
    rows, cols = strat.shape
    data_row, data_col = data.shape
    #Layer thickness, time to form, env type
    layers = pd.DataFrame(0, index=range(rows*cols), columns=['Layer_Thickness', 'Layer_Time', 'Lobe', 'Channel', 'Total_Dep', 'Total_Time', 'Stasis_Proportion', 'Deposition_Proportion'], dtype = float)

    #starting position
    pos = 0
    for index in range(rows):
        for col in range(cols):
            #0 layers are contained in the data at the ends of some rows
            if strat.at[index, col] == 0:
                None
            else:
                #targets
                layers.at[pos, 'Layer_Thickness'] = strat.at[index, col]
                layers.at[pos, 'Layer_Time'] = time.at[index, col]
                layers.at[pos, 'Lobe'] = lobes.at[index, col]
                layers.at[pos, 'Channel'] = channels.at[index, col]

                #indexing patch
                if (col+1 >= cols) or (start.at[index, col+1] == 0):
                    end = data_col-1
                else:
                    end = start.at[index, col+1]

                #first position for each layer events
                data_pos = int(start.at[index, col])
                stasis_count = 0   
                dep_count = 0 
                count = 0
                #for all events in a given time frame
                for x in range(int(end - start.at[index, col])):
                    #increase total deposition
                    if data.at[index, data_pos] > data.at[index, data_pos-1]:
                        layers.at[pos,'Total_Dep'] += data.at[index, data_pos] - data.at[index, data_pos-1]
                        dep_count += 1

                    #add more time of stasis
                    if data.at[index, data_pos] == data.at[index, data_pos-1]:
                        stasis_count += 1

                    #total time increase and next event
                    count += 1
                    data_pos += 1
                #divide by 0 fix
                if count == 0:
                    count = 1
                #add total time behind each layer and proportion of stasis to total time
                layers.at[pos,'Total_Time'] = count*2
                layers.at[pos,'Stasis_Proportion'] = stasis_count / count
                layers.at[pos,'Deposition_Proportion'] = dep_count / count
                #next line
                pos += 1
    return layers
################################################################################################################################

#flatten 3D matrices by column to row
def flatten(data):
    rows, cols, depth = data.shape

    # Initialize an empty list to collect rows
    flattened_rows = []

    # Iterate over each position in the xy plane
    for i in range(rows):
        for j in range(cols):
            # Collect all values along the z axis for each (i, j) position
            flattened_rows.append(data[i, j, :])

    # Convert the list to a 2D numpy array
    matrix_2d = np.array(flattened_rows)

    return pd.DataFrame(matrix_2d)
################################################################################################################################

#load mat file as Pandas Dataframe
def load_mat_data(file, key):
    mat = scipy.io.loadmat(file)
    data = mat[key]
    return np.array(data)

#################################################################################################################################     
    
data = load_mat_data("ZdataD_final.mat", 'ZD')
lobes = load_mat_data("lobes.mat", 'New_Lobes')
channels = load_mat_data("channel.mat", 'channel')

data = flatten(data)
lobes = flatten(lobes)
channels = flatten(channels)

print("Flattened")

strat, time, start_times, lobes, channels = current_strat(data, lobes, channels)

# Convert data to DataFrames or Series
strat_df = pd.DataFrame(strat, columns=['strat'])
time_df = pd.DataFrame(time, columns=['time'])
start_times_df = pd.DataFrame(start_times, columns=['start_times'])
lobes_df = pd.DataFrame(lobes, columns=['lobes'])
channels_df = pd.DataFrame(channels, columns=['channels'])

# Save each variable to an individual CSV file
strat_df.to_csv('strat.csv', index=False)
time_df.to_csv('time.csv', index=False)
start_times_df.to_csv('start_times.csv', index=False)
lobes_df.to_csv('lobes.csv', index=False)
channels_df.to_csv('channels.csv', index=False)

print("Data saved to CSV files successfully!")

layer_stats = layer_info(strat, time, start_times, data, lobes, channels)

print("We up in this bitch")
layer_stats.to_csv("Layer_Stats.csv", sep=',')


