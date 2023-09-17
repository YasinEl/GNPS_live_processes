import pyopenms as oms
import pandas as pd
import argparse


def calculate_tic(file_path):
    # Initialize the MSExperiment instance
    exp = oms.MSExperiment()
    oms.MzMLFile().load(file_path, exp)
    
    # Calculate TIC and RT for each MS1 spectrum and store in lists
    tic_values = [spectrum.calculateTIC() for spectrum in exp if spectrum.getMSLevel() == 1]
    rt_values = [spectrum.getRT() for spectrum in exp if spectrum.getMSLevel() == 1]

    # Create a dataframe
    df = pd.DataFrame({
        'rt': rt_values,
        'intensity': tic_values
    })

    return df


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="create inventory table for MS2 scans.")
    parser.add_argument('--file_path', type=str, help="Path to the mzML file.")
    
    args = parser.parse_args()
    file_path = args.file_path

    df = calculate_tic(file_path)

    df.to_csv('MS1_inventory_table.csv', index=False)


