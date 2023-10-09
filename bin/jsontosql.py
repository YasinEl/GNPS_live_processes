import argparse
import json
import pandas as pd
from sqlalchemy import create_engine

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
                    'name': metric_name
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





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create inventory table for MS2 scans.")
    parser.add_argument('--json_path', type=str, help="Path to the aggregated json.")
    args = parser.parse_args()


    df_untargeted = create_filtered_table(args.json_path, type_ = 'single')
    df_untargeted.drop(['name'], axis=1, inplace=True)

    df_untargeted = df_untargeted.groupby(['mzml_file', 'date_time']).agg(lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else None).reset_index()

    date_time_mapping = df_untargeted.groupby('mzml_file')['date_time'].first()
    df_untargeted['date_time'] = df_untargeted['mzml_file'].map(date_time_mapping)
    df_untargeted.sort_values(by='date_time', inplace=True)
    df_untargeted['datetime_order'] = df_untargeted['date_time'].rank(method='min').astype(int)

   
    df_targeted = create_filtered_table(args.json_path, type_ = 'standards')
    df_targeted.drop(['date_time'], axis=1, inplace=True)


    
    engine = create_engine('sqlite:///aggregated_summary.db')

    df_untargeted.to_sql('untargetedSummary', engine, if_exists='replace')
    df_targeted.to_sql('targetedSummary', engine, if_exists='replace')



