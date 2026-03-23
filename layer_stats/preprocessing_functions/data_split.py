import pandas as pd 
import numpy as np
import time
import scipy.io

#Takes in depth at each time
#Returns ending stratigraphy, time to form each layer, and start time of each layer
def current_strat(data, env_data):
    rows, cols = data.shape
    #current strat
    strat = pd.DataFrame(0, index=range(rows), columns=range(cols), dtype = float)
    #layer time to form
    time = pd.DataFrame(0, index=range(rows), columns=range(cols), dtype = float)
    #starting time of each layer
    start_times = pd.DataFrame(0, index=range(rows), columns=range(cols), dtype = float)
    # layer environment
    env = pd.DataFrame(5, index=range(rows), columns=range(cols), dtype = int)

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
                #env
                env.at[index, layer_index] = env_data.at[index,col]
                new_start = False

            #deposition
            if data.at[index, col] > prev_change:
                #update layer thickness, update time taken to form
                strat.at[index, layer_index] += (data.at[index, col] - prev_change)
                time.at[index, layer_index] += 1

                #env
                env.at[index, layer_index] = env_data.at[index,col]
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
                        #env
                        env.at[index, layer_index] = 5
                        erosion = 0

                    #too much erosion for given layer, spill to next layer
                    elif erosion >= strat.at[index, layer_index]:
                        #zero out that layer, and move to next layer
                        erosion -= strat.at[index, layer_index]
                        strat.at[index, layer_index] = 0
                        time.at[index, layer_index] = 0
                        start_times.at[index, layer_index] = 0
                        #env
                        env.at[index, layer_index] = 5

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
    env.drop(columns=zero_columns, inplace=True)

    #set column names starting at 0, for indexing in next step
    new_column_names = list(range(len(strat.columns)))
    strat.columns = new_column_names
    time.columns = new_column_names
    start_times.columns = new_column_names
    env.columns = new_column_names

    return strat, time, start_times, env