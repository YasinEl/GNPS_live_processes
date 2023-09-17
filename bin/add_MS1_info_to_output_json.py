import json
import pandas as pd
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add MS2 info to json.")
    parser.add_argument('--output_json_path', type=str, help="Path to the json file.")
    parser.add_argument('--ms1_inv_csv', type=str, help="Paths to MS1 inventory csv.")


    args = parser.parse_args()

    with open(args.output_json_path, 'r') as file:
        output_json = json.load(file)


    df_MS1 = pd.read_csv(args.ms1_inv_csv)

    metric = {
        "name": "MS1_inventory",
        "type": "MS1_inventory",
        "collection": "MS1_inventory",
        "reports": {
            "MS1_spectra": len(df_MS1),
            "TIC_sum": df_MS1['intensity'].sum(),
            "TIC_median": df_MS1['intensity'].median(),
            "MS1_inventory": df_MS1.to_dict()
        }
    }


    output_json['metrics'].append(metric)

    with open('mzml_summary.json', 'w') as file:
        json.dump(output_json, file, indent=4)