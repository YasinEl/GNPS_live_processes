import sys
import os
import pandas as pd
from handle_parameter_file import prepare_parameter_file
from io import StringIO


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



if __name__ == "__main__":



    parameter_file_path = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/parameter file.xlsx"
    feature_adducts_file_path = "/home/yasin/yasin/projects/GNPS_live_processes/nf_output/features_adducts.csv"

    parameter_dict = prepare_parameter_file(parameter_file_path)
    df_features = read_custom_csv(feature_adducts_file_path)

    print(df_features.head(1))

    print(parameter_dict['standards'])


