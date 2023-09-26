import json
import pandas as pd

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
                                row.update(value)
                            else:
                                row[key] = value
                
                table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Identify columns containing lists
    list_columns = [col for col in df.columns if df[col].apply(isinstance, args=(list,)).any()]

    # Explode the DataFrame based on list columns
    if list_columns:
        # Explode all list columns
        for col in list_columns:
            df = df.explode(col)
        df.reset_index(drop=True, inplace=True)
    
    return df

