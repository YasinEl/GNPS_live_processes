import json
import pandas as pd
import argparse
from io import StringIO
import numpy as np
from collections import Counter
import statistics

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
    parser.add_argument('--feature_csv', type=str, help="Paths to feature csv.")
    parser.add_argument('--ms2_inv_csv', type=str, help="Paths to MS2 inventory csv.")
    args = parser.parse_args()

    with open(args.output_json_path, 'r') as file:
        output_json = json.load(file)

    df_features = read_custom_csv(args.feature_csv)
    df_ms2 = pd.read_csv(args.ms2_inv_csv)

    #get triggered features
    detected_features = set(df_features['label'])
    triggered_features = set(df_ms2['feature_label'].dropna())

    # Initialize 'MS2_grade' column with 'vacant' where 'feature_label' is None
    df_ms2['MS2_grade'] = None
    # Set 'MS2_grade' to 'missing' only if it's not already set
    condition_missing = (~df_ms2['feature_label'].isna()) & (df_ms2['MS Level'].isna()) & (df_ms2['MS2_grade'].isna())
    df_ms2.loc[condition_missing, 'MS2_grade'] = 'missing'

    # Set 'MS2_grade' to 'vacant' only if it's not already set
    condition_vacant = df_ms2['feature_label'].isna() & df_ms2['MS2_grade'].isna()
    df_ms2.loc[condition_vacant, 'MS2_grade'] = 'vacant'

    # Find the index with the highest 'Feature Apex Intensity' for each 'feature_label'
    idx_max_intensity = df_ms2[df_ms2['feature_label'].notna()].groupby(['feature_label', 'Collision energy'])['Feature Apex intensity'].idxmax()

    # Set 'MS2_grade' to 'successful' for these indices, only if it's not already set
    condition_successful = df_ms2.index.isin(idx_max_intensity) & df_ms2['MS2_grade'].isna()
    df_ms2.loc[condition_successful, 'MS2_grade'] = 'successful'

    # Set 'MS2_grade' to 'redundant' where it's still None
    df_ms2['MS2_grade'].fillna('redundant', inplace=True)


    
    df_ms2['successful_count'] = None
    df_ms2['redundant_count'] = None
    df_ms2['vacant_count'] = None

    # Initialize lists to store unique row IDs for each count type
    #successful_rows = []
    redundant_rows = []
    vacant_rows = []
    vacant_mz = []
    no_scans = []
    successful_rows = []
    added_vacant_rows = set()
    # Loop through rows where MS2_grade is 'missing'
    for idx, row in df_ms2[df_ms2['MS2_grade'] == 'missing'].iterrows():
        rt = row['Retention Time (min)']
        fwhm = row['FWHM']
        
        # Find rows where rt is within the range and MS2_grade is not 'missing'
        condition = (df_ms2['Retention Time (min)'] >= rt - fwhm) & (df_ms2['Retention Time (min)'] <= rt + fwhm) & (df_ms2['MS2_grade'] != 'missing')
        filtered_rows = df_ms2.loc[condition]
        
        # Count occurrences of each grade
        counts = filtered_rows['MS2_grade'].value_counts()
        
        # Update the counts in the DataFrame
        df_ms2.loc[idx, 'successful_count'] = counts.get('successful', 0)
        df_ms2.loc[idx, 'redundant_count'] = counts.get('redundant', 0)
        df_ms2.loc[idx, 'vacant_count'] = counts.get('vacant', 0)
        
        # Update the unique row ID lists
        successful_rows.extend(filtered_rows[filtered_rows['MS2_grade'] == 'successful'].index.tolist())
        redundant_rows.extend(filtered_rows[filtered_rows['MS2_grade'] == 'redundant'].index.tolist())
        vacant_rows.extend(filtered_rows[filtered_rows['MS2_grade'] == 'vacant'].index.tolist())

        # Update vacant_mz list with unique rows
        vacant_filtered_rows = filtered_rows[(filtered_rows['MS2_grade'] == 'vacant') & (~filtered_rows.index.isin(added_vacant_rows))]
        vacant_mz_values = vacant_filtered_rows['Precursor m/z'].round(2).tolist()
        vacant_mz.extend(vacant_mz_values)


        if counts.get('redundant', 0) == 0 and counts.get('vacant', 0) == 0:
            no_scans.append('1')

        added_vacant_rows.update(vacant_filtered_rows.index.tolist())

    # Remove duplicates from the lists
    successful_rows = list(set(successful_rows))
    redundant_rows = list(set(redundant_rows))
    vacant_rows = list(set(vacant_rows))

    vacantMZ_count_table = Counter(vacant_mz)
        
    total_count = len(vacant_mz)
    less_than_5_count = 0
    more_than_5_count = 0

    for value, count in vacantMZ_count_table.items():
        if count < 5:
            less_than_5_count += count
        elif count > 5:
            more_than_5_count += count

    percentage_less_than_5 = (less_than_5_count / total_count) * 100
    percentage_more_than_5 = (more_than_5_count / total_count) * 100



    #get lowest triggered features
    df_ms2_filtered = df_ms2[
    (df_ms2['MS2_grade'] != 'vacant') &
    (df_ms2['Purity'] >= 0.9) &
    (df_ms2['Peak count (filtered)'] >= 4)
    ]

    MS2_quality_percentile = np.percentile(df_ms2_filtered['Feature Apex intensity'],5)

    df_missed_features_above_percentile = df_ms2[
        (df_ms2['MS2_grade'] != 'vacant') &
        (df_ms2['Feature Apex intensity'] > MS2_quality_percentile) &
        (df_ms2['MS Level'].isna())
    ]

    df_missed_features_below_percentile = df_ms2[
        (df_ms2['MS2_grade'] != 'vacant') &
        (df_ms2['Feature Apex intensity'] < MS2_quality_percentile) &
        (df_ms2['MS Level'].isna())
    ]
    ####
    #Are we collecting MS2s for enough features
    ####
    #   percentage of features with MS2             -> do (DONE)
    #   intensity below which we have no MS2        -> outlier detection for MS2 precursor intensity and take lowest within distribution or low percentile. (DONE)
    #   why do we not have enough MS2
    #       -wasting time
    #           -baseline monitoring                -> x most common redundant triggers (DONE)
    #           -multiple MS2 per feature           -> (DONE)
    #       -MS2 scan rate (as a fraction of FWHM)  -> (DONE)
    #       -no MS2 get triggered
    #           -MS2 AGC too high
    #           -not enough high features
    #           
    #percentage of MS2-features of sufficient quality

    metric = {
        "name": "feature_inventory",
        "type": "single",
        "collection": "feature_inventory",
        "reports": {
            "Feature_count": len(df_features),
            "Triggered_features": (len(triggered_features) / len(detected_features)) * 100,
            "Triggered_lowest_int_percentile": MS2_quality_percentile,
            "Features_above_percentile": len(df_features[df_features['intensity'] >= MS2_quality_percentile]),
            "Features_below_percentile": len(df_features[df_features['intensity'] < MS2_quality_percentile]),
            "Missed_triggers_above_percentile": len(df_missed_features_above_percentile),
            "Missed_triggers_below_percentile": len(df_missed_features_below_percentile),
            "Obstacles_above_percentile_redundant_scans": len(redundant_rows),
            "Obstacels_above_percentile_vacant_scans": len(vacant_rows),
            "Obstacels_above_percentile_successfull_scans": len(successful_rows),
            "Obstacels_above_percentile_No_scans": len(no_scans),
            "Percentage_of_vacant_obstacles_more_than_5": percentage_more_than_5,
            "Feature_intensities_Q1": np.percentile(df_features['intensity'],25),
            "Feature_intensities_Q2": np.percentile(df_features['intensity'],50),
            "Feature_intensities_Q3": np.percentile(df_features['intensity'],75),
            "Feature_FWHM_Q1": np.percentile(df_features['FWHM'],25),
            "Feature_FWHM_Q2": np.percentile(df_features['FWHM'],50),
            "Feature_FWHM_Q3": np.percentile(df_features['FWHM'],75)

        }
    }

    print(metric)

    output_json['metrics'].append(metric)

    with open('mzml_summary.json', 'w') as file:
        json.dump(output_json, file, indent=4)
