import pyopenms as oms
import pandas as pd
import argparse

def filter_values(values, factor=3):
    min_val = min(values)
    threshold = factor * min_val
    return [val for val in values if val >= threshold]


def is_within_ppm(observed_mz, expected_mz, ppm_tolerance):
    ppm_difference = ((observed_mz - expected_mz) / expected_mz) * 1e6
    return abs(ppm_difference) <= ppm_tolerance


def add_unique_mz_col(df, ppm_val = 10):
    ppm = ppm_val * 1e-6
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
        
    parser = argparse.ArgumentParser(description="create inventory table for MS2 scans.")
    parser.add_argument('--file_path', type=str, help="Path to the mzML file.")
    
    args = parser.parse_args()

    file_path = args.file_path

    #file_path = '/home/yasin/yasin/projects/GNPS_live_processes/random_data/sixmix.mzML'

    # Initialize the run instance
    exp = oms.MSExperiment()
    oms.MzMLFile().load(file_path, exp)

    # Create a list to hold spectra
    data = []

    prev_MS1_spectrum = None
    for spectrum in exp:
        if spectrum.getMSLevel() == 1:
            prev_MS1_spectrum = spectrum

        elif spectrum.getMSLevel() > 1 and prev_MS1_spectrum:

            ms_level = spectrum.getMSLevel()
            rt = spectrum.getRT() / 60.0  # Convert to minutes
            precursor = spectrum.getPrecursors()[0]
            precursor_mz = precursor.getMZ()
            precursor_int = precursor.getIntensity()
            precursor_charge = precursor.getCharge()

            mzs, ints = spectrum.get_peaks()

            peaks_count = len(ints)
            peaks_count_filtered = len(filter_values(ints, 3))


            #get MS2 purity
            lower_offset = precursor.getIsolationWindowLowerOffset()
            higher_offset = precursor.getIsolationWindowUpperOffset()
            precursor_mzs, precursor_ints = prev_MS1_spectrum.get_peaks()
            isolated_mzs = [mz for mz in precursor_mzs if (precursor_mz - lower_offset) <= mz <= (precursor_mz + higher_offset)]
            isolated_ints = [intensity for mz, intensity in zip(precursor_mzs, precursor_ints) if (precursor_mz - lower_offset) <= mz <= (precursor_mz + higher_offset)]
            precursor_intensity = next((intensity for mz, intensity in zip(precursor_mzs, precursor_ints) if is_within_ppm(mz, precursor_mz, 2)), 0)
            total_intensity = sum(isolated_ints)
            purity = precursor_intensity / total_intensity if total_intensity else 0


            
            data.append([ms_level, rt, precursor_mz, precursor_charge, precursor_int, purity, peaks_count, peaks_count_filtered])



    df = pd.DataFrame(data, columns=['MS Level', 'Retention Time (min)', 'Precursor m/z', 'Precursor charge', 'Precursor intensity', 'Purity', 'Peak count', 'Peak count (filtered)'])
    
    df = add_unique_mz_col(df)
    
    df.to_csv('MS2_inventory_table.csv', index = False)
