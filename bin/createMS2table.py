import pyopenms as oms
import pandas as pd
import argparse
from bisect import bisect_left, bisect_right

def filter_values(values, factor=3):
    min_val = min(values)
    threshold = factor * min_val
    return [val for val in values if val >= threshold]

def is_within_ppm(observed_mz, expected_mz, ppm_tolerance):
    ppm_difference = ((observed_mz - expected_mz) / expected_mz) * 1e6
    return abs(ppm_difference) <= ppm_tolerance

def add_unique_mz_col(df, ppm_val=10):
    ppm = ppm_val * 1e-6
    df = df.sort_values(by='Precursor m/z').reset_index(drop=True)
    
    # Initial group identifier
    group_id = 1
    
    # Placeholder for the group assignments
    groups = [-1] * len(df)
    
    i = 0
    while i < len(df):
        mz = df.loc[i, 'Precursor m/z']
        
        # Calculate the lower and upper bounds for ppm
        lower_bound = mz - (mz * ppm)
        upper_bound = mz + (mz * ppm)
        
        # Filtering rows that have 'Precursor m/z' within the ppm range of the current mz
        filtered = df[(df['Precursor m/z'] >= lower_bound) & (df['Precursor m/z'] <= upper_bound)]
        
        # Assign the current group identifier to all rows in this group
        for idx in filtered.index:
            groups[idx] = group_id
        
        group_id += 1  # Increment the group identifier
        i = filtered.index[-1] + 1  # Skip to the next mz outside of the current ppm range
    
    # Assign the group values to a new column in the original dataframe
    df['Group'] = groups
    return df 

# Pre-sort the features by their m/z value
def preprocess_features(feature_map):
    sorted_features = sorted([(feature.getMZ(), feature) for feature in feature_map])
    
    # Create time bins for features
    time_bins = {}
    for feature in feature_map:
        rt_feature = feature.getRT() / 60.0  # Convert to minutes
        bin_id = int(rt_feature // 1)  # 1 minute bins
        if bin_id not in time_bins:
            time_bins[bin_id] = []
        time_bins[bin_id].append(feature)
    
    return sorted_features, time_bins

if __name__ == '__main__':
    # Argument parsing (no change here)
    parser = argparse.ArgumentParser(description="Create inventory table for MS2 scans.")
    parser.add_argument('--file_path', type=str, help="Path to the mzML file.")
    parser.add_argument('--featureXML_path', type=str, help="Path to the featureXML file.")
    args = parser.parse_args()
    file_path = args.file_path
    featureXML_path = args.featureXML_path

    # Load data 
    feature_map = oms.FeatureMap()
    oms.FeatureXMLFile().load(featureXML_path, feature_map)
    exp = oms.MSExperiment()
    oms.MzMLFile().load(file_path, exp)

    # Pre-process features
    sorted_features, time_bins = preprocess_features(feature_map)
    
    data = []
    prev_MS1_spectrum = None
    
    for spectrum in exp:
        if spectrum.getMSLevel() == 1:
            prev_MS1_spectrum = spectrum

        elif spectrum.getMSLevel() == 2 and prev_MS1_spectrum:

            ms_level = spectrum.getMSLevel()
            rt = spectrum.getRT() /60
            precursor = spectrum.getPrecursors()[0]
            precursor_mz = precursor.getMZ()
            precursor_charge = precursor.getCharge()

            mzs, ints = spectrum.get_peaks()

            # Find the intensity of the most intense fragment peak
            max_fragment_intensity = max(ints) if ints.size > 0 else 0

            # Initialize as not associated with any feature
            associated_with_feature = False
            apex_intensity = 0

            # Counting peaks
            peaks_count = len(ints)
            peaks_count_filtered = len(filter_values(ints, 3))

            # Calculate purity
            lower_offset = precursor.getIsolationWindowLowerOffset()
            higher_offset = precursor.getIsolationWindowUpperOffset()
            precursor_mzs, precursor_ints = prev_MS1_spectrum.get_peaks()
            isolated_ints = []
            isolated_mzs = []
            total_intensity = 0
            precursor_int = 0  # Initialize as 0, if no value meets the condition, it will remain 0

            for mz, intensity in zip(precursor_mzs, precursor_ints):
                if (precursor_mz - lower_offset) <= mz <= (precursor_mz + higher_offset):
                    isolated_ints.append(intensity)
                    isolated_mzs.append(mz)
                    if is_within_ppm(mz, precursor_mz, 15):
                        precursor_int = float(intensity)  # Assuming that only one mz will meet this condition


            total_intensity = sum(isolated_ints)
            purity = precursor_int / total_intensity if total_intensity > 0 else None

            associated_feature_label = "N/A"
            temporal_distance_score = None
            apex_intensity_2 = None
            prec_apex_ratio = None

            # Find the relevant time bins to search in
            relevant_bin_ids = [int((rt // 1) + i) for i in range(-1, 2)]  # Search in neighboring bins as well
            relevant_features = [feature for bin_id in relevant_bin_ids for feature in time_bins.get(bin_id, [])]
            
            # Perform binary search on the sorted features
            low = bisect_left(sorted_features, (precursor_mz - precursor_mz * 15e-6, None))
            high = bisect_right(sorted_features, (precursor_mz + precursor_mz * 15e-6, None))
            candidates = [feature for mz, feature in sorted_features[low:high]]
            
            # Only loop through the features that are both in the time bin and m/z range
            for feature in [x for x in candidates if x in relevant_features]:

                mz_feature = feature.getMZ()
                rt_feature = feature.getRT() / 60.0 

                convex_hull = feature.getConvexHulls()[0]  # Assuming one convex hull per feature
                points = convex_hull.getHullPoints()
                
                rt_values = [point[0] for point in points]
                start_time_feature = min(rt_values) / 60
                end_time_feature = max(rt_values) / 60


                #start_time_feature = feature.getConvexHull().getHullPoints()[0][0] / 60.0  # First point, RT
                #end_time_feature = feature.getConvexHull().getHullPoints()[-1][0] / 60.0  # Last point, RT

                if is_within_ppm(mz_feature, precursor_mz, 15) and rt > start_time_feature and rt < end_time_feature:
                    associated_feature_label = feature.getUniqueId()

                    apex_intensity = feature.getMetaValue("max_height")
                    feature_FWHM = feature.getMetaValue("FWHM")

                    
                    #apex_intensity = feature.getIntensity()

                    # Calculate the temporal distance score
                    #feature_duration = end_time_feature - start_time_feature
                    #temporal_distance_score = (rt - rt_feature) / feature_duration if feature_duration else 0
                    temporal_distance_score = (rt - rt_feature) * 60
                    prec_apex_ratio = precursor_int/apex_intensity
                    break



            data.append([ms_level, rt, precursor_mz, precursor_charge, max_fragment_intensity, precursor_int, purity, peaks_count, peaks_count_filtered, associated_feature_label, apex_intensity, feature_FWHM, temporal_distance_score, prec_apex_ratio])

    df = pd.DataFrame(data, columns=['MS Level', 'Retention Time (min)', 'Precursor m/z', 'Precursor charge', 'Max fragment intensity', 'Precursor intensity', 'Purity', 'Peak count', 'Peak count (filtered)', 'Associated Feature Label', 'Feature Apex intensity', 'FWHM', 'Rel Feature Apex distance', 'Prec-Apex intensity ratio'])

    df = add_unique_mz_col(df)
    
    df.to_csv('MS2_inventory_table.csv', index=False)
