import os
import pandas as pd
from io import StringIO
import re
import argparse
import json


def create_regex(df):
    results = []

    # Group by 'set'
    groups = df.groupby('set')

    for set_val, group in groups:
        patterns_to_include = []
        patterns_to_exclude = []

        for _, row in group.iterrows():
            if row['parameters'] == '#STD_set_pattern' and row['pattern'] != '*':
                patterns_to_include.append(row['pattern'])
            elif row['parameters'] == '#STD_set_skip':
                patterns_to_exclude.append(row['pattern'])

        # Construct the regex pattern
        if patterns_to_include:
            include_pattern = '(?=.*(' + '|'.join(patterns_to_include) + '))'
        else:
            include_pattern = ''

        exclude_pattern = '(?!.*(' + '|'.join(patterns_to_exclude) + '))' if patterns_to_exclude else ''

        regex_pattern = '^' + include_pattern + exclude_pattern + '.*$'

        results.append(pd.DataFrame({'set': [set_val], 'regex': [regex_pattern]}))

    return pd.concat(results, ignore_index=True)


def prepare_parameter_file(file_path, mzml_path):

    df_STD = pd.read_excel(file_path, sheet_name='Standards per Set', usecols=range(7), engine='openpyxl')

    df_params = pd.read_excel(file_path, sheet_name='Filenames to Set Mapping', usecols=range(3), engine='openpyxl')
    df_params = df_params.dropna(subset=['parameters'])

    df_samplename_set = df_params[df_params['parameters'].isin(['#STD_set_pattern', '#STD_set_skip'])]
    df_regex = create_regex(df_samplename_set)

    name_of_mzml = os.path.splitext(mzml_path)[0]
    
    # Check which standard sets should be extracted and filter standards to relevant ones
    STD_sets_to_extract = []
    for _, row in df_regex.iterrows():
        pattern = row['regex']
        if re.search(pattern, name_of_mzml):
            STD_sets_to_extract.append(row['set'])

    df_STD = df_STD[df_STD['set'].isin(STD_sets_to_extract)]

    params_dict = {'df_standards': df_STD,
                   'df_params': df_params,
                   'df_regex': df_regex}
       
    return params_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process an excel file to create regex patterns based on set mappings.')
    parser.add_argument('--file_path', type=str, help='Path to the excel file.')
    parser.add_argument('--mzml_path', type=str, help='Path to the mzML file.')

    args = parser.parse_args()

    param_dict = prepare_parameter_file(args.file_path, args.mzml_path)

    print(param_dict)

    with open('prepared_parameters.json', 'w') as f:
        json_data = {key: value.to_json() for key, value in param_dict.items()}
        json.dump(json_data, f, indent=4)

    


