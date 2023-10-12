import argparse
import json
import os
from datetime import datetime
import pandas as pd
import re

def valid_datetime(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid date-time: '{s}'.")

def extract_timestamp(file_path):
    pattern = r'startTimeStamp="([\d\-\:T]+)Z"'
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            match = re.search(pattern, line)
            if match:
                timestamp_str = match.group(1)
                dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
    return ''

def main():
    parser = argparse.ArgumentParser(description="Prepare a json for collecting GNPS outputs and add mzml file name.")
    parser.add_argument('--filename', type=str, help="Path to the mzML file.")
    parser.add_argument('--param_json', type=str, help="Path to the prepared parameter json file.")
    parser.add_argument('--datetime', type=valid_datetime, help="Time of upload.")

    args = parser.parse_args()
    
    # Initialize variables
    QC_type = '-'
    is_blank = False



    base_name = os.path.basename(args.filename)
    
    timestamp = args.datetime.strftime('%Y-%m-%d %H:%M:%S')
    timestamp = extract_timestamp(args.filename)

    if timestamp == '':
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        

    with open(args.param_json, 'r') as file:
        param_dict = json.load(file)

    df_params_json = param_dict.get('df_params')
    df_params = pd.read_json(df_params_json)


    # Filter rows and check for substrings in 'base_name'
    qc_rows = df_params[df_params['parameters'] == '#QC_set_pattern']
    for pattern in qc_rows['pattern']:
        if pattern.lower() in base_name.lower():
            QC_type = pattern
            break

    solvent_rows = df_params[df_params['parameters'] == '#solventblank_set_pattern']
    for pattern in solvent_rows['pattern']:
        if pattern in base_name:
            is_blank = True
            break


    
    output_data = {'mzml_name': base_name,
                   'time_of_upload': timestamp}
    
    output_data['metrics'] = []

    metric = {
    "name": "Sample_metadata",
    "type": "single",
    "collection": "Sample_metadata",
    "reports": {
        "QC_type": QC_type,
        "is_blank": is_blank
        }
        }

    output_data['metrics'].append(metric)


    
    with open('mzml_summary.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()
