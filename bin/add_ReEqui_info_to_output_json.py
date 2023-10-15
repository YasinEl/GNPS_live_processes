import json
import pandas as pd
import argparse
from io import StringIO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add MS1 info to json.")
    parser.add_argument('--output_json_path', type=str, help="Path to the json file.")
    parser.add_argument('--reEqui_csv_path', type=str, help="Paths to ReEquilibration csv.")
    args = parser.parse_args()

    with open(args.output_json_path, 'r') as file:
        output_json = json.load(file)

    df_reE = pd.read_csv(args.reEqui_csv_path)



    metric = {
        "name": "ReEquilibration_inventory",
        "type": "single",
        "collection": "ReEquilibration_inventory",
        "reports": {
            "ReEquilibration_inventory": df_reE.to_dict()
        }
    }

    output_json['metrics'].append(metric)

    with open('mzml_summary.json', 'w') as file:
        json.dump(output_json, file, indent=4)
