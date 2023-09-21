import json
import os
import argparse
import pandas as pd

def fill_missing_values(group):
    single_row = group[group['type'] == 'single'].iloc[0]
    return group.fillna(single_row)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Aggreagte mzml jsons summaries into a table")
    parser.add_argument('--json_directory', type=str, help="Path to json directory..")
    
    args = parser.parse_args()

    data_list = []

    # Loop through each JSON file in the directory
    for filename in os.listdir(args.json_directory):
        filepath = os.path.join(args.json_directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.json'):
            with open(filepath, 'r') as file:
                content = json.load(file)
                
                # Extract information for each metric and append to data_list
                for metric in content['metrics']:
                    data_dict = {
                        "mzml_name": content["mzml_name"],
                        "datetime": content["time_of_upload"],
                        "name": metric["name"],
                        "type": metric["type"],
                        "collection": metric["collection"],
                    }
                    data_dict.update({k: v for k, v in metric["reports"].items() if not isinstance(v, (dict, list))})

                    data_list.append(data_dict)

    # Convert data_list into a pandas DataFrame
    df = pd.DataFrame(data_list)

    #df = df.groupby(['mzml_name', 'datetime']).apply(fill_missing_values).reset_index(drop=True)


    df.to_csv('all_jsons_table.csv', index = False)