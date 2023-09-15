import pyopenms as oms
import pandas as pd

def assign_group(df):
    ppm = 10e-6
    df = df.sort_values(by='Precursor m/z').reset_index(drop=True)
    
    # Initial group identifier
    group_id = 1
    
    # Placeholder for the group assignments
    groups = [-1] * len(df)
    
    i = 0
    while i < len(df):
        mz = df.loc[i, 'Precursor m/z']
        
        # Calculate the lower and upper bounds for 10 ppm
        lower_bound = mz - (mz * ppm)
        upper_bound = mz + (mz * ppm)
        
        # Filtering rows that have 'Precursor m/z' within the 10 ppm range of the current mz
        filtered = df[(df['Precursor m/z'] >= lower_bound) & (df['Precursor m/z'] <= upper_bound)]
        
        # Assign the current group identifier to all rows in this group
        for idx in filtered.index:
            groups[idx] = group_id
        
        group_id += 1  # Increment the group identifier
        i = filtered.index[-1] + 1  # Skip to the next mz outside of the current 10 ppm range
    
    # Assign the group values to a new column in the original dataframe
    df['Group'] = groups
    return df

            

if __name__ == '__main__':

    file_path = 'work/7d/5d64cedcf2835e6805d121207db795/MS2_inventory_table.csv'


    df = pd.read_csv(file_path)

    df = assign_group(df)

    print(df.head(3))

    

    