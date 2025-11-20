import pandas as pd

def error_check(data):
    rows, cols = data.shape

    for index in range(rows):
        # Start with the first column as the initial elevation reference
        elev = data.iloc[index, 0]

        # Loop through each column, starting from the second column
        for col in range(1, cols):
            if abs(data.iloc[index, col] - elev) >= 0.5:
                # If change exceeds threshold, update 'elev' to the current value
                elev = data.iloc[index, col]
            else:
                # Otherwise, set the current cell to the previous 'elev' value
                data.iloc[index, col] = elev

    return data
