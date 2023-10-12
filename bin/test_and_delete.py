import json
import pandas as pd
import argparse
from io import StringIO
import numpy as np



def create_filtered_table(json_file, name=None, type_=None, collection=None, include_keys=None):
    with open(json_file, 'r') as f:
        data = json.load(f)

    table_data = []

    for entry in data:
        mzml_name = entry.get('mzml_name', None)
        time_of_upload = entry.get('time_of_upload', None)
        metrics = entry.get('metrics', [])

        for metric in metrics:
            metric_name = metric.get('name', None)
            metric_type = metric.get('type', None)
            metric_collection = metric.get('collection', None)

            if ((name is None or name == metric_name) and
                    (type_ is None or type_ == metric_type) and
                    (collection is None or collection == metric_collection)):

                row = {
                    'mzml_file': mzml_name,
                    'date_time': time_of_upload,
                    'name': metric_name,
                    'type': metric_type,
                    'collection': metric_collection
                }

                reports = metric.get('reports', {})

                for key, value in reports.items():
                    if include_keys is None:
                        if not isinstance(value, (list, dict)):
                            row[key] = value
                    else:
                        if key in include_keys:
                            if isinstance(value, dict):
                                if isinstance(list(value.values())[0], dict):
                                    for key2, value2 in value.items():
                                        row[key2] = list(value2.values())
                                        if not list(value2.keys())[0][0].isdigit():
                                            row['variable'] = list(value2.keys())

                                else:
                                    row.update(value)
                            else:
                                row[key] = value

                table_data.append(row)

    df = pd.DataFrame(table_data)

    # Identify columns containing lists
    list_columns = [col for col in df.columns if df[col].apply(isinstance, args=(list,)).any()]

    # Explode the DataFrame based on list columns
    if list_columns:
        df = df.explode(list_columns)
        df.reset_index(drop=True, inplace=True)

    return df



json_path = '/home/yasin/yasin/projects/GNPS_live_processes/nf_output/mzml_summary_aggregation.json'

df = create_filtered_table(json_path, collection='MS1_inventory', include_keys='highest_30_by_quarter_RT')

date_time_mapping = df.groupby('mzml_file')['date_time'].first()
df['date_time'] = df['mzml_file'].map(date_time_mapping)
df.sort_values(by='date_time', inplace=True)
df['datetime_order'] = df['date_time'].rank(method='min').astype(int)

df_meta = create_filtered_table(json_path, collection='Sample_metadata')
df_meta = df_meta[df_meta['QC_type'] != "-"]

df = pd.merge(df, df_meta[['mzml_file', 'QC_type']], on='mzml_file', how='inner')


output_list = []

for qc_type in df['QC_type'].unique():
    qc_df = df[df['QC_type'] == qc_type].copy()
    qc_df.sort_values(by='datetime_order', inplace=True)
    unique_dt_orders = qc_df['datetime_order'].unique()
    
    for i in range(len(unique_dt_orders) - 1):
        order1 = unique_dt_orders[i]
        order2 = unique_dt_orders[i + 1]
        
        current_dt_df = qc_df[qc_df['datetime_order'] == order1]
        next_dt_df = qc_df[qc_df['datetime_order'] == order2]
        
        # Combine all rows into a single table for each DataFrame
        current_combined_table = pd.concat(current_dt_df['highest_30_by_quarter_RT'].apply(pd.DataFrame).tolist(), ignore_index=True)
        next_combined_table = pd.concat(next_dt_df['highest_30_by_quarter_RT'].apply(pd.DataFrame).tolist(), ignore_index=True)
        
        output_dict = {
            'qctype': qc_type,
            'mzml1': current_dt_df['mzml_file'].iloc[0],
            'order1': order1,
            'mzml2': next_dt_df['mzml_file'].iloc[0],
            'order2': order2,
            'rt_bin_0': None,
            'rt_bin_1': None,
            'rt_bin_2': None,
            'int_bin_0': None,
            'int_bin_1': None,
            'int_bin_2': None
        }
        
        for third in [0, 1, 2]:
            rt_diffs = []
            int_diffs = []
            
            current_third_table = current_combined_table[current_combined_table['third'] == third]
            next_third_table = next_combined_table[next_combined_table['third'] == third]
            
            for idx, row in current_third_table.iterrows():
                matching_rows = next_third_table[
                    (next_third_table['mz'].round(2) == row['mz'].round(2)) &
                    (abs(next_third_table['rt'] - row['rt']) <= 15)
                ]
                
                if len(matching_rows) > 1:
                    matching_rows['rt_diff'] = abs(matching_rows['rt'] - row['rt'])
                    best_match = matching_rows.loc[matching_rows['rt_diff'].idxmin()]
                    rt_diffs.append(best_match['rt_diff'])
                    best_match = best_match.copy()
                    best_match['int_diff_percentage'] = 100 * (best_match['intensity'] - row['intensity']) / row['intensity']
                    int_diffs.append(best_match['int_diff_percentage'])
                elif len(matching_rows) == 1:
                    rt_diffs.append(abs(matching_rows['rt'].iloc[0] - row['rt']))
                    matching_rows = matching_rows.copy()
                    matching_rows['int_diff_percentage'] = 100 * (matching_rows['intensity'] - row['intensity']) / row['intensity']
                    int_diffs.append(matching_rows['int_diff_percentage'])
            
            rt_key = f'rt_bin_{third}'
            int_key = f'int_bin_{third}'
            output_dict[rt_key] = np.median(rt_diffs) if len(rt_diffs) >= 3 else None
            output_dict[int_key] = np.median(int_diffs) if len(int_diffs) >= 3 else None
        
        output_list.append(output_dict)

df = pd.DataFrame(output_list)



# Create a new DataFrame with selected columns
new_df = df[['qctype', 'order2', 'mzml2', 'rt_bin_0', 'rt_bin_1', 'rt_bin_2', 'int_bin_0', 'int_bin_1', 'int_bin_2']].copy()

# Rename the columns
new_df.columns = ['qctype', 'order', 'mzml', 'rt_bin_0', 'rt_bin_1', 'rt_bin_2', 'int_bin_0', 'int_bin_1', 'int_bin_2']

# Add the first file of each qctype as a new row with None values for the 6 variables
first_rows = df.drop_duplicates('qctype')[['qctype', 'order1', 'mzml1']].copy()
first_rows.columns = ['qctype', 'order', 'mzml']
first_rows[['rt_bin_0', 'rt_bin_1', 'rt_bin_2', 'int_bin_0', 'int_bin_1', 'int_bin_2']] = None

# Concatenate the new DataFrame with the first rows
final_df = pd.concat([new_df, first_rows], ignore_index=True)

# Sort the DataFrame
final_df.sort_values(['qctype', 'order'], inplace=True, ascending=[False, True])
final_df.reset_index(drop=True, inplace=True)


print(final_df)