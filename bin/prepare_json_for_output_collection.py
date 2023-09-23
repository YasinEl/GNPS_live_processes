import argparse
import json
import os
from datetime import datetime

def valid_datetime(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid date-time: '{s}'.")



def main():
    parser = argparse.ArgumentParser(description="Prepare a json for collecting GNPS outputs and add mzml file name.")
    parser.add_argument('--filename', type=str, help="Path to the mzML file.")
    parser.add_argument('--datetime', type=valid_datetime, help="Time of upload.")
    
    args = parser.parse_args()

    timestamp = args.datetime.strftime('%Y-%m-%d %H:%M:%S')

    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    base_name = os.path.basename(args.filename)
    
    output_data = {'mzml_name': base_name,
                   'time_of_upload': timestamp}
    output_data['metrics'] = []
    
    with open('mzml_summary.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    main()
