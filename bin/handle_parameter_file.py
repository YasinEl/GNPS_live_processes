import pandas as pd
import re
import argparse


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


def prepare_parameter_file(file_path):

    df_STD = pd.read_excel(file_path, sheet_name='Standards per Set', usecols=range(5))

    df_params = pd.read_excel(file_path, sheet_name='Filenames to Set Mapping', usecols=range(3))
    df_params = df_params.dropna(subset=['parameters'])

    df_samplename_set = df_params[df_params['parameters'].isin(['#STD_set_pattern', '#STD_set_skip'])]
    df_regex = create_regex(df_samplename_set)

    params_dict = {'df_standards': df_STD,
                   'df_params': df_params,
                   'df_regex': df_regex}
    
    return params_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process an excel file to create regex patterns based on set mappings.')
    parser.add_argument('file_path', type=str, help='Path to the excel file.')
    
    args = parser.parse_args()
    prepare_parameter_file(args.file_path)


