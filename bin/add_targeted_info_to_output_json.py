import json
import pandas as pd
from io import StringIO
import os
import argparse
import xmltodict
        
def read_custom_csv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Find the line index with '#FEATURE'
        header_index = next(i for i, line in enumerate(lines) if '#FEATURE' in line)

        # Collect lines from '#FEATURE' until a line starting with '#' is encountered
        data_lines = [lines[header_index]]  # Start with the header line
        for line in lines[header_index+1:]:
            if line.strip().startswith('#'):
                continue
            data_lines.append(line)

        # Convert this subset of lines into a DataFrame
        df = pd.read_csv(StringIO(''.join(data_lines)), header=0)  # set header=0 to use the first line as the header

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add output of targeted OpenMS analysis to json.")
    parser.add_argument('--output_json_path', type=str, help="Path to the mzML file.")
    parser.add_argument('--std_sets', type=str, help="Paths to std_set csv files.")
    
    args = parser.parse_args()


    #output_json_path = '/home/yasin/yasin/projects/GNPS_live_processes/nf_output/mzml_summary.json'
    #std_sets = ['/home/yasin/yasin/projects/GNPS_live_processes/nf_output/set_1.csv', '/home/yasin/yasin/projects/GNPS_live_processes/nf_output/set_2.csv']


    with open(args.output_json_path, 'r') as file:
        output_json = json.load(file)

    std_sets = args.std_sets.split(",")

    for csv_path in std_sets:
        
        #construct df from xml
        with open(csv_path, 'r') as file:
            content = file.read()
            data = xmltodict.parse(content)

        features = data['featureMap']['featureList']['feature']
        
        rows = []
        eics = []

        if int(data['featureMap']['featureList']['@count']) > 1:
            for feature in features:
                #collect_single_value_data
                user_params = feature['UserParam'] + feature['subordinate']['feature'][0]['UserParam']
                row_data = {item['@name']: item['@value'] for item in user_params}
                rows.append(row_data)

                #collect EICs
                eic_data = feature['subordinate']['feature'][0]['convexhull']["pt"]

                eic_data_prep = {
                    "rt": [float(item["@x"]) for item in eic_data],
                    "intensity": [float(item["@y"]) for item in eic_data]
                    }
                eics.append(eic_data_prep)


        elif int(data['featureMap']['featureList']['@count']) == 1:
            user_params = features['UserParam'] + features['subordinate']['feature'][0]['UserParam']
            row_data = {item['@name']: item['@value'] for item in user_params}
            rows.append(row_data)

            #collect EICs
            eic_data = features['subordinate']['feature'][0]['convexhull']["pt"]

            eic_data_prep = {
                "rt": [float(item["@x"]) for item in eic_data],
                "intensity": [float(item["@y"]) for item in eic_data]
                }
            eics.append(eic_data_prep)


        df = pd.DataFrame(rows)


        # Extract the basename without the extension for the collection name
        collection_name = os.path.splitext(os.path.basename(csv_path))[0]

        # For each row in the dataframe
        for idx, row in df.iterrows():
            metric = {
                "name": str(row["label"]),
                "type": "standards",
                "collection": collection_name,
                "reports": {
                    "MZ": float(row['MZ']),
                    "RT": float(row['peak_apex_position']),
                    "Height": float(row['peak_apex_int']),
                    "FWHM": float(row['width_at_50']),
                    "EIC": eics[idx]
                }
            }
            output_json['metrics'].append(metric)

with open('mzml_summary.json', 'w') as file:
    json.dump(output_json, file, indent=4)
