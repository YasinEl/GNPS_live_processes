import argparse
import json
import os
from pyteomics import mzml

def count_ms2_scans(filename):
    count = 0
    with mzml.read(filename) as spectra:
        for spectrum in spectra:
            if spectrum['ms level'] == 2:
                count += 1
    return count

def write_json_to_stdout(data):
    print(json.dumps(data, indent=4))

def main():
    parser = argparse.ArgumentParser(description="Count the number of MS2 scans in an mzML file.")
    parser.add_argument('filename', type=str, help="Path to the mzML file.")
    
    args = parser.parse_args()
    
    num_ms2_scans = count_ms2_scans(args.filename)
    
    base_name = os.path.basename(args.filename)
    
    output_data = {'mzml_name': base_name, 'MS2_count': num_ms2_scans}
    write_json_to_stdout(output_data)

    #print(f"Number of MS2 scans in {args.filename}: {num_ms2_scans}")
    #print(f"MS2 count written to {output_filename}")

if __name__ == "__main__":
    main()
