import argparse
import json
import os

def write_json_to_stdout(data):
    print(json.dumps(data, indent=4))

def main():
    parser = argparse.ArgumentParser(description="Prepare a json for collecting GNPS outputs and add mzml file name.")
    parser.add_argument('filename', type=str, help="Path to the mzML file.")
    
    args = parser.parse_args()
        
    base_name = os.path.basename(args.filename)
    
    output_data = {'mzml_name': base_name}
    write_json_to_stdout(output_data)

if __name__ == "__main__":
    main()
