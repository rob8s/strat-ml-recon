import pandas as pd

def tag(df):
    # Add the first new column 'High_Erosion' where the condition (Total_Dep - Layer_Thickness >= 6.5) is met
    df['High_Erosion'] = (df['Total_Dep'] - df['Layer_Thickness'] >= 6.5).astype(int)

    # Add the second feature where (Stasis_Proportion * Total_Time) exceeds 115
    #df['High_Stasis'] = (df['Stasis_Proportion'] * df['Total_Time'] >= 115).astype(int)

    return df