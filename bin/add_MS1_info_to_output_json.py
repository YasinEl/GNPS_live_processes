import json
import pandas as pd
import argparse
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

    parser = argparse.ArgumentParser(description="Add MS2 info to json.")
    parser.add_argument('--output_json_path', type=str, help="Path to the json file.")
    parser.add_argument('--ms1_inv_csv', type=str, help="Paths to MS1 inventory csv.")
    parser.add_argument('--feature_csv', type=str, help="Paths to feature csv.")

    args = parser.parse_args()

    with open(args.output_json_path, 'r') as file:
        output_json = json.load(file)

    df_MS1 = pd.read_csv(args.ms1_inv_csv)

    df_features = read_custom_csv(args.feature_csv)

    metric = {
        "name": "MS1_inventory",
        "type": "MS1_inventory",
        "collection": "MS1_inventory",
        "reports": {
            "MS1_spectra": len(df_MS1),
            "TIC_sum": df_MS1['intensity'].sum(),
            "TIC_max": df_MS1['intensity'].max(),
            "TIC_median": df_MS1['intensity'].median(),
            "MS1_Features": len(df_features),
            "MS1_inventory": df_MS1.to_dict()
        }
    }


    output_json['metrics'].append(metric)

    with open('mzml_summary.json', 'w') as file:
        json.dump(output_json, file, indent=4)