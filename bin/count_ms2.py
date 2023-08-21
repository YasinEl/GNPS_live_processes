import argparse
from pyteomics import mzml

def count_ms2_scans(filename):
    count = 0
    with mzml.read(filename) as spectra:
        for spectrum in spectra:
            if spectrum['ms level'] == 2:
                count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description="Count the number of MS2 scans in an mzML file.")
    parser.add_argument('filename', type=str, help="Path to the mzML file.")
    
    args = parser.parse_args()
    
    num_ms2_scans = count_ms2_scans(args.filename)
    print(f"Number of MS2 scans in {args.filename}: {num_ms2_scans}")

if __name__ == "__main__":
    main()
