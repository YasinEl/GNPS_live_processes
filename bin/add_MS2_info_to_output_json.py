import json
import pandas as pd
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add MS2 info to json.")
    parser.add_argument('--output_json_path', type=str, help="Path to the json file.")
    parser.add_argument('--ms2_inv_csv', type=str, help="Paths to MS2 inventory csv.")


    args = parser.parse_args()

    with open(args.output_json_path, 'r') as file:
        output_json = json.load(file)


    df_MS2 = pd.read_csv(args.ms2_inv_csv)

    metric = {
        "name": "MS2_inventory",
        "type": "MS2_inventory",
        "collection": "MS2_inventory",
        "reports": {
            "MS2_spectra": len(df_MS2['MS Level'].dropna()),
            "Unique_prec_MZ": int(df_MS2['Group'].dropna().nunique()),
            "Median_Peak_count": int(df_MS2['Peak count'].dropna().median()),
            "Median_filtered_Peak_count": int(df_MS2['Peak count (filtered)'].dropna().median()),
            "Median_precuror_purity": df_MS2['Purity'].dropna().median(),
            "MS2_inventory": df_MS2.to_dict()
        }
    }


    output_json['metrics'].append(metric)

    with open('mzml_summary.json', 'w') as file:
        json.dump(output_json, file, indent=4)