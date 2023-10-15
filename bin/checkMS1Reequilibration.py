from pyopenms import MSExperiment, MzMLFile
import pandas as pd
import numpy as np
from lxml import etree
import argparse

def get_table_comparing_start_and_end_of_injection(file_path, massRangeMin, massRangeMax, nr_of_bins = 10):
    # Initialize MzMLFile and MSExperiment objects
    mzml = MzMLFile()
    exp = MSExperiment()

    # Load the mzML file
    mzml.load(file_path, exp)

    # Filter out MS1 spectra
    ms1_spectra = [s for s in exp if s.getMSLevel() == 1]

    # Take the 20th to 40th and the last 20 MS1 scans
    early_scans = ms1_spectra[19:40]
    late_scans = ms1_spectra[-20:]

    # Initialize lists to store summed intensities
    early_bins = [0]*nr_of_bins
    late_bins = [0]*nr_of_bins

    # Fetch lower and upper mass range boundaries
    tree = etree.parse(file_path)
    root = tree.getroot()
    namespaces = {'ns': 'http://psi.hupo.org/ms/mzml'}

    if massRangeMin > 0 and massRangeMin > 0:
        lower_limit = massRangeMin
        upper_limit = massRangeMax
    else: 
        lower_limit = float(root.find(".//ns:cvParam[@accession='MS:1000501']", namespaces=namespaces).get('value'))
        upper_limit = float(root.find(".//ns:cvParam[@accession='MS:1000500']", namespaces=namespaces).get('value'))

    # Calculate bin width
    bin_width = (upper_limit - lower_limit) / nr_of_bins

    # Function to sum intensities in bins
    def sum_in_bins(scans, bins, lower_limit, bin_width):
        for scan in scans:
            mz, intensity = scan.get_peaks()
            for m, i in zip(mz, intensity):
                if m >= lower_limit and m <= upper_limit:
                    bin_idx = int((m - lower_limit) // bin_width)
                    bins[bin_idx] += i

    # Sum intensities in bins
    sum_in_bins(early_scans, early_bins, lower_limit, bin_width)
    sum_in_bins(late_scans, late_bins, lower_limit, bin_width)

    # Create DataFrame
    df = pd.DataFrame({
        'InjectionScans': early_bins,
        'ReequilibratedScans': late_bins,
        'BinID': np.arange(1, nr_of_bins+1),
        'LowerMz': [round(lower_limit + i*bin_width, 2) for i in range(nr_of_bins)],
        'UpperMz': [round(lower_limit + (i+1)*bin_width, 2) for i in range(nr_of_bins)]
    })

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create inventory table for MS2 scans.")
    parser.add_argument('--file_path', type=str, help="Path to the mzML file.")
    parser.add_argument('--massRangeMin', type=float, help="Minimum limit of mz range")
    parser.add_argument('--massRangeMax', type=float, help="maximum limit of mz range")
    
    args = parser.parse_args()
    file_path = args.file_path
    df = get_table_comparing_start_and_end_of_injection(file_path, massRangeMin=args.massRangeMin, massRangeMax=args.massRangeMax, nr_of_bins=20)
    
    df.to_csv('reequilibration_check.csv', index=False)