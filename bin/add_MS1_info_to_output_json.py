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
    new_col_names = {}
    for col in df.columns:
        if 'MZbin ' in col and '-' in col:
            lower, upper = col.split(' ')[-1].split('-')
            mz_bins.append((float(lower), float(upper)))
            new_col_names[col] = f'MZbin {len(mz_bins)}'
    df.rename(columns=new_col_names, inplace=True)
    return df, mz_bins

def calculate_feature_metrics(df, mz_bins):
    feature_metrics = {}
    for i, (lower, upper) in enumerate(mz_bins, 1):
        df_bin = df[(df['mz'] >= lower) & (df['mz'] <= upper)]
        feature_metrics[f"MZbin {i}"] = {
            "Feature median int": df_bin['intensity'].median(),
            "Feature median FWHM": df_bin['FWHM'].median(),
            "Feature count": len(df_bin)
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
        col = f'MZbin {len(tic_metrics) + 1}'
        if col in df_MS1.columns:
            #tic_metrics[f"{lower}-{upper}"] = {
            tic_metrics[col] = {
                "TIC_sum": df_MS1[col].sum(),
                "TIC_max": df_MS1[col].max(),
                "TIC_median": df_MS1[col].median()
            }

    feature_metrics = calculate_feature_metrics(df_features, mz_bins)


    df_features = df_features[['rt', 'mz', 'intensity', 'FWHM']]

    ######
    # get highest 30 per 33% of rt
    ######
    rt_max = df_MS1['rt'].max()

    highest_30_by_third = []
    for i in range(3):

        #split table
        lower_bound = i * 0.33 * rt_max
        upper_bound = (i + 1) * 0.33 * rt_max
        if lower_bound < 30 and upper_bound > 30:
            lower_bound = 30
        new_table = df_features[(df_features['rt'] >= lower_bound) & (df_features['rt'] < upper_bound)]

        #FWHM filtering
        median_fwhm = new_table['FWHM'].median()
        new_table = new_table[(new_table['FWHM'] >= 0.6 * median_fwhm) & (new_table['FWHM'] <= 1.4 * median_fwhm)]

        #remove things with more than 1 peak
        new_table['mz'] = new_table['mz'].round(2)
        new_table = new_table.drop_duplicates(subset='mz', keep=False)

        #remove potential adducts
        new_table = new_table.sort_values('intensity', ascending=False)
        to_keep = []
        for idx, row in new_table.iterrows():
            if all(abs(row['rt'] - kept_row['rt']) >= 0.5 for kept_row in to_keep):
                to_keep.append(row)
        
        #archive
        new_table = pd.DataFrame(to_keep).nlargest(30, 'intensity')
        new_table.reset_index(drop=True, inplace=True)
        new_table['third'] = i
        highest_30_by_third.append(new_table.to_dict())


    metric = {
        "name": "MS1_inventory",
        "type": "single",
        "collection": "MS1_inventory",
        "reports": {
            "MS1_spectra": len(df_MS1),
            "MS1_Features": len(df_features),
            "TIC_sum": df_MS1['all MZbins'].sum(),
            "TIC_max": df_MS1['all MZbins'].max(),
            "TIC_median": df_MS1['all MZbins'].median(),
            "first_MS1_scan_rt": df_MS1['rt'].min(),
            "last_MS1_scan_rt": df_MS1['rt'].max(),
            "TIC_bins": mz_bins,
            "TIC_metrics": tic_metrics,
            "Feature_metrics": feature_metrics,
            "highest_30_by_quarter_RT": highest_30_by_third,
            "MS1_inventory": df_MS1.to_dict()
        }
    }

    output_json['metrics'].append(metric)

    with open('mzml_summary.json', 'w') as file:
        json.dump(output_json, file, indent=4)
