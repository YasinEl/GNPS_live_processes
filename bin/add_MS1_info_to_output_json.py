import json
import pandas as pd
import argparse
from io import StringIO

def read_custom_csv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        header_index = next(i for i, line in enumerate(lines) if '#FEATURE' in line)
        data_lines = [lines[header_index]]
        for line in lines[header_index+1:]:
            if line.strip().startswith('#'):
                continue
            data_lines.append(line)
        df = pd.read_csv(StringIO(''.join(data_lines)), header=0)
    return df

def rename_csv(df):
    mz_bins = []
    #new_col_names = {}
    for col in df.columns:
        if 'TIC_intensity' in col and '-' in col:
            lower, upper = col.split('_')[-1].split('-')
            mz_bins.append((float(lower), float(upper)))
            #new_col_names[col] = f'TIC_intensity_{len(mz_bins)}'
    #df.rename(columns=new_col_names, inplace=True)
    return df, mz_bins

def calculate_feature_metrics(df, mz_bins):
    feature_metrics = {}
    for lower, upper in mz_bins:
        df_bin = df[(df['mz'] >= lower) & (df['mz'] <= upper)]
        feature_metrics[f"{lower}-{upper}"] = {
            "Feature median int": df_bin['intensity'].median(),
            "Feature median FWHM": df_bin['FWHM'].median(),
            "Feature count": len(df_bin)#,
            #"intensity_sum": df_bin['intensity'].sum(),
            #"intensity_max": df_bin['intensity'].max()
        }
    return feature_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add MS1 info to json.")
    parser.add_argument('--output_json_path', type=str, help="Path to the json file.")
    parser.add_argument('--ms1_inv_csv', type=str, help="Paths to MS1 inventory csv.")
    parser.add_argument('--feature_csv', type=str, help="Paths to feature csv.")
    args = parser.parse_args()

    with open(args.output_json_path, 'r') as file:
        output_json = json.load(file)

    df_MS1 = pd.read_csv(args.ms1_inv_csv)
    df_MS1, mz_bins = rename_csv(df_MS1)
    df_features = read_custom_csv(args.feature_csv)

    tic_metrics = {}
    for lower, upper in mz_bins:
        col = f'TIC_intensity_{len(tic_metrics) + 1}'
        if col in df_MS1.columns:
            tic_metrics[f"{lower}-{upper}"] = {
                "TIC_sum": df_MS1[col].sum(),
                "TIC_max": df_MS1[col].max(),
                "TIC_median": df_MS1[col].median()
            }

    feature_metrics = calculate_feature_metrics(df_features, mz_bins)

    metric = {
        "name": "MS1_inventory",
        "type": "single",
        "collection": "MS1_inventory",
        "reports": {
            "MS1_spectra": len(df_MS1),
            "MS1_Features": len(df_features),
            "TIC_sum": df_MS1['TIC_intensity_complete'].sum(),
            "TIC_max": df_MS1['TIC_intensity_complete'].max(),
            "TIC_median": df_MS1['TIC_intensity_complete'].median(),
            #"TIC_bins": mz_bins,
            "TIC_metrics": tic_metrics,
            "Feature_metrics": feature_metrics,
            "MS1_inventory": df_MS1.to_dict()
        }
    }

    output_json['metrics'].append(metric)

    with open('mzml_summary.json', 'w') as file:
        json.dump(output_json, file, indent=4)
