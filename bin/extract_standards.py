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

def find_closest_mz(mz_value, df_features, ppm):
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
    name_of_mzml = 'Pool.mzML'
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

    # A new dictionary to hold the results with sorted keys
    result_dict = {key: chemical_dict[key] for key in df_standards.columns}

    # For each mz in the chemical_dict, find the closest mz in df_features and update the dictionary
    for idx, mz in enumerate(chemical_dict['mz']):
        matched = find_closest_mz(mz, df_features, ppm = ppm)
        if matched:
            for key, value in matched.items():
                detected_key = "detected_" + key
                if detected_key not in result_dict:
                    result_dict[detected_key] = [None]*idx
                result_dict[detected_key].append(value)

    # To ensure all lists in the dictionary have the same length, fill in None values
    max_length = max(len(v) for v in result_dict.values())
    for key, values in result_dict.items():
        if len(values) < max_length:
            result_dict[key].extend([None] * (max_length - len(values)))

    print(result_dict)
