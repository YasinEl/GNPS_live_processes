import os
import json
import argparse

def aggregate_json_files(dir_path):
    aggregated_data = []

    files = os.listdir(dir_path)

    for file_name in files:
        if file_name.endswith('.json'):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
            aggregated_data.append(data)
    return aggregated_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate JSON files into one.')
    parser.add_argument('--json_directory', type=str, help='Path to the directory containing JSON files.')
    args = parser.parse_args()

    aggregated_data = aggregate_json_files(args.json_directory)

    print('test')

    with open('mzml_summary_aggregation.json', 'w') as f:
        json.dump(aggregated_data, f, indent=4)
