import pandas as pd 
import numpy as np
import time
import scipy.io
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
    
    # Create a DataFrame for better visualization (optional)
    df = pd.DataFrame(matrix_2d)
    
    return df