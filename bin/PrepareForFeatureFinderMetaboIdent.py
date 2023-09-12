import pandas as pd
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create tsv tables for OpenMS (one per STD set).')
    parser.add_argument('--parameter_json', type=str, help='Path to parameter json.')

    args = parser.parse_args()

    with open(args.parameter_json, 'r') as f:
        json_data = json.load(f)
        parameter_dict = {key: pd.read_json(value) for key, value in json_data.items()}

    df_standards = parameter_dict['df_standards']

    column_rename_map = {
    'name': 'CompoundName',
    'formula': 'SumFormula',
    'retention time [seconds]': 'RetentionTime',
    'charge': 'Charge'
    }

    df_STD_openms = df_standards.copy()
    
    df_STD_openms.rename(columns=column_rename_map, inplace=True)
    df_STD_openms = df_STD_openms[['set', 'CompoundName', 'SumFormula', 'Charge', 'RetentionTime']]

    df_STD_openms['Mass'] = 0
    df_STD_openms['RetentionTimeRange'] = 0
    df_STD_openms['IsoDistribution'] = 0

    # Desired order of columns
    column_order = ['set', 'CompoundName', 'SumFormula', 'Mass', 'Charge', 'RetentionTime', 'RetentionTimeRange', 'IsoDistribution']

    # Reordering the columns
    df_STD_openms = df_STD_openms[column_order]
  
    # Grouping by 'set' and iterating over each group
    for set_val, group in df_STD_openms.groupby('set'):
        # Dropping the 'set' column
        group_without_set = group.drop('set', axis=1)
        
        # Saving to .tsv
        filename = f"set_{set_val}.tsv"
        group_without_set.to_csv(filename, sep='\t', index=False)

