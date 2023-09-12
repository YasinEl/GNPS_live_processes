import sys
import os
import pandas as pd
from handle_parameter_file import prepare_parameter_file
from io import StringIO
import re

def read_custom_csv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Find the line index with '#FEATURE'
        header_index = next(i for i, line in enumerate(lines) if '#FEATURE' in line)

        # Collect lines from '#FEATURE' until a line starting with '#' is encountered
        data_lines = [lines[header_index]]  # Start with the header line
        for line in lines[header_index+1:]:
            if line.strip().startswith('#'):
                continue
            data_lines.append(line)

        # Convert this subset of lines into a DataFrame
        df = pd.read_csv(StringIO(''.join(data_lines)), header=0)  # set header=0 to use the first line as the header

    return df

def find_closest_mz(mz_value, df_features, ppm = 5):
    lower_bound = mz_value - (ppm * 1e-6 * mz_value)
    upper_bound = mz_value + (ppm * 1e-6 * mz_value)
    
    filtered = df_features[(df_features['mz'] >= lower_bound) & (df_features['mz'] <= upper_bound)]
    
    if not filtered.empty:
        # Return the row with the highest intensity as a dictionary
        return filtered.loc[filtered['intensity'].idxmax()].to_dict()
    else:
        return None

if __name__ == "__main__":

    parameter_file_path = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/parameter file.xlsx"
    feature_adducts_file_path = "/home/yasin/yasin/projects/GNPS_live_processes/nf_output/features_adducts.csv"
    name_of_mzml = 'sixmix.mzML'
    ppm = 5

    # Prepare parameter file
    parameter_dict = prepare_parameter_file(parameter_file_path)
    df_standards = parameter_dict['df_standards']
    df_regex = parameter_dict['df_regex']

    name_of_mzml = os.path.splitext(name_of_mzml)[0]
    
    # Check which standard sets should be extracted and filter standards to relevant ones
    STD_sets_to_extract = []
    for _, row in df_regex.iterrows():
        pattern = row['regex']
        if re.search(pattern, name_of_mzml):
            STD_sets_to_extract.append(row['set'])

    df_standards = df_standards[df_standards['set'].isin(STD_sets_to_extract)]
    chemical_dict = df_standards.to_dict(orient='list')
    df_features = read_custom_csv(feature_adducts_file_path)

    result_dict = {}

    for idx, mz in enumerate(chemical_dict['mz']):
        set_val = chemical_dict['set'][idx]
        if set_val not in result_dict:
            result_dict[set_val] = {}

        input_data = {key: chemical_dict[key][idx] for key in chemical_dict.keys()}
        
        matched = find_closest_mz(mz, df_features)
        if matched:
            extracted_data = matched
        else:
            extracted_data = {} # or any default value you prefer if no match is found

        mz_data = {
            'input_data': input_data,
            'extracted_data': extracted_data
        }

        result_dict[set_val][f"mz_{idx}"] = mz_data

    import pprint
    pprint.pprint(result_dict)
