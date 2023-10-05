import pyopenms as oms
import pandas as pd
import argparse
import statistics
from xml.etree import ElementTree as ET

def get_MS1_MZ_range(mzml_path):
    # Initialize lists to store limits
    lower_limits = []
    upper_limits = []

    # Load mzML file
    tree = ET.parse(mzml_path)
    root = tree.getroot()

    # Define namespace to navigate XML tree
    namespaces = {'ns': 'http://psi.hupo.org/ms/mzml'}

    # Iterate through each spectrum
    for spectrum in root.findall(".//ns:spectrum", namespaces=namespaces):
        ms_level = spectrum.find(".//ns:cvParam[@accession='MS:1000511']", namespaces=namespaces)
        
        if ms_level is not None and ms_level.attrib['value'] == "1":
            lower_limit = spectrum.find(".//ns:cvParam[@accession='MS:1000501']", namespaces=namespaces)
            upper_limit = spectrum.find(".//ns:cvParam[@accession='MS:1000500']", namespaces=namespaces)
            
            if lower_limit is not None:
                lower_limits.append(float(lower_limit.attrib['value']))
            
            if upper_limit is not None:
                upper_limits.append(float(upper_limit.attrib['value']))

    # Calculate median
    if lower_limits:
        median_lower_limit = statistics.median(lower_limits)
    else:
        median_lower_limit = None

    if upper_limits:
        median_upper_limit = statistics.median(upper_limits)
    else:
        median_upper_limit = None

    return [median_lower_limit, median_upper_limit]

def calculate_bins(mz_limits, total_windows=3, min_bin_size=50, division=5, equal_bins=False):
    if total_windows % 2 == 0 and equal_bins == False:
        raise ValueError("Number of windows must be odd.")

    mz_range = mz_limits[1] - mz_limits[0]

    bin_size = mz_range / division

    while bin_size < min_bin_size:
        division -= 2
        bin_size = mz_range / division
        if division <= 1:
            break

    if equal_bins:
        bins = []
        for i in range(total_windows):
            lower = mz_limits[0] + i * bin_size
            upper = mz_limits[0] + (i + 1) * bin_size
            bins.append((lower, upper))
        return bins
    
    edge_windows = total_windows // 2
    section_size = division // 3
    edge_bin_size = mz_range * (section_size / division)
    middle_bin_size = mz_range - 2 * edge_bin_size

    bins = []
    for i in range(edge_windows):
        lower = mz_limits[0] + i * edge_bin_size / edge_windows
        upper = mz_limits[0] + (i + 1) * edge_bin_size / edge_windows
        bins.append((lower, upper))

    middle_lower = mz_limits[0] + edge_bin_size
    middle_upper = middle_lower + middle_bin_size
    bins.append((middle_lower, middle_upper))

    for i in range(edge_windows):
        lower = mz_limits[1] - (edge_windows - i) * edge_bin_size / edge_windows
        upper = mz_limits[1] - (edge_windows - i - 1) * edge_bin_size / edge_windows
        bins.append((lower, upper))
    
    return bins



def calculate_tic(file_path):
    exp = oms.MSExperiment()
    oms.MzMLFile().load(file_path, exp)
    
    rt_values = []
    tic_values = []
    
    mz_limits = get_MS1_MZ_range(file_path)
    bin_limits = calculate_bins(mz_limits, total_windows=3, min_bin_size=50, division=3, equal_bins=True)
    tic_values_bins = [[] for _ in range(len(bin_limits))]

    for spectrum in exp:
        if spectrum.getMSLevel() == 1:
            tic = spectrum.calculateTIC()
            rt = spectrum.getRT()

            rt_values.append(rt)
            tic_values.append(tic)

            mz, intensity = spectrum.get_peaks()
            for i, (lower, upper) in enumerate(bin_limits):
                mask = (mz >= lower) & (mz < upper)
                tic_bin = sum(intensity[mask])
                tic_values_bins[i].append(tic_bin)
                
    column_names = {f'TIC_MZbin_{lower:.2f}-{upper:.2f}': tic_bin for i, tic_bin in enumerate(tic_values_bins)
                    for lower, upper in [bin_limits[i]]}
    
    df = pd.DataFrame({
        'rt': rt_values,
        'TIC_MZ_complete': tic_values,
        **column_names
    })
    
    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create inventory table for MS2 scans.")
    parser.add_argument('--file_path', type=str, help="Path to the mzML file.")
    args = parser.parse_args()
    file_path = args.file_path
    df = calculate_tic(file_path)
    df.to_csv('MS1_inventory_table.csv', index=False)
