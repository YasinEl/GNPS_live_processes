import json
import pandas as pd
from io import StringIO
import os
import argparse
import xmltodict
        
def read_custom_tsv(file_path):
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
    parser.add_argument('--output_json_path', type=str, help="Path to the json file.")
    parser.add_argument('--std_sets', type=str, help="Paths to std_set featureXML files.")
    parser.add_argument('--std_sets_tsvs', type=str, help="Paths to std_set tsv files.")
    
    args = parser.parse_args()


    with open(args.output_json_path, 'r') as file:
        output_json = json.load(file)

    std_sets = args.std_sets.split(",")

    if len(std_sets) == 1 and os.path.splitext(os.path.basename(std_sets[0]))[0] == 'set_none':
        with open('mzml_summary.json', 'w') as file:
            json.dump(output_json, file, indent=4)
        
        exit()

    std_sets_tsvs = args.std_sets_tsvs.split(",")

    for featureXML_path, tsv_path in zip(std_sets, std_sets_tsvs):
        
        #construct df from xml
        with open(featureXML_path, 'r') as file:
            content = file.read()
            data = xmltodict.parse(content)

        features = data.get('featureMap', {}).get('featureList', {}).get('feature', None)

        
        rows = []
        eics = []

        if int(data['featureMap']['featureList']['@count']) > 1 and features is not None:
            for feature in features:
                #collect_single_value_data
                user_params = feature['UserParam'] + feature['subordinate']['feature'][0]['UserParam']
                row_data = {item['@name']: item['@value'] for item in user_params}
                rows.append(row_data)

                #collect EICs
                eic_data = feature['subordinate']['feature'][0]['convexhull']["pt"]

                # Convert to float and pair rt and intensity
                eic_data_prep = list(zip(
                    [float(item["@x"]) for item in eic_data],
                    [float(item["@y"]) for item in eic_data]
                ))

                # Sort by intensity in descending order
                eic_data_prep.sort(key=lambda x: x[1], reverse=True)

                # Take the top 30 highest intensity values
                eic_data_prep = eic_data_prep[:30]

                # Sort by rt in ascending order
                eic_data_prep.sort(key=lambda x: x[0])

                # Separate rt and intensity back into their own lists
                eic_data_filtered = {
                    "rt": [round(item[0],2) for item in eic_data_prep],
                    "intensity": [round(item[1],0) for item in eic_data_prep]
                }

                eics.append(eic_data_filtered)



        elif int(data['featureMap']['featureList']['@count']) == 1 and features is not None:
            user_params = features['UserParam'] + features['subordinate']['feature'][0]['UserParam']
            row_data = {item['@name']: item['@value'] for item in user_params}
            rows.append(row_data)

            #collect EICs
            eic_data = features['subordinate']['feature'][0]['convexhull']["pt"]

            # Convert to float and pair rt and intensity
            eic_data_prep = list(zip(
                [float(item["@x"]) for item in eic_data],
                [float(item["@y"]) for item in eic_data]
            ))

            # Sort by intensity in descending order
            eic_data_prep.sort(key=lambda x: x[1], reverse=True)

            # Take the top 30 highest intensity values
            eic_data_prep = eic_data_prep[:30]

            # Sort by rt in ascending order
            eic_data_prep.sort(key=lambda x: x[0])

            # Separate rt and intensity back into their own lists
            eic_data_filtered = {
                "rt": [round(item[0],2) for item in eic_data_prep],
                "intensity": [int(item[1]) for item in eic_data_prep]
            }

            eics.append(eic_data_filtered)



        
        # Extract the basename without the extension for the collection name
        collection_name = os.path.splitext(os.path.basename(featureXML_path))[0]

        if features is not None:
            df = pd.DataFrame(rows)

            # For each row in the dataframe
            for idx, row in df.iterrows():
                metric = {
                    "name": str(row["label"]),
                    "type": "standards",
                    "collection": collection_name,
                    "reports": {
                        "MZ": round(float(row['MZ']),4),
                        "RT": round(float(row['peak_apex_position']),0),
                        "Height": row['peak_apex_int'],
                        "FWHM": float(row['width_at_50']),
                        "EIC": eics[idx]
                    }
                }
                output_json['metrics'].append(metric)

        # Identify the tsv file that belongs to the current std_set
        matching_tsv = next((tsv for tsv in std_sets_tsvs if os.path.splitext(os.path.basename(tsv))[0] == collection_name), None)

        if matching_tsv:
            tsv_df = pd.read_csv(matching_tsv, delimiter='\t')

            # Identify missing labels in df
            if features is not None:
                missing_labels = set(tsv_df['CompoundName']) - set(df['label'])
            else:
                missing_labels = set(tsv_df['CompoundName'])

            # Add missing labels to df and output_json
            for missing_label in missing_labels:
                metric = {
                    "name": str(missing_label),
                    "type": "standards",
                    "collection": collection_name,
                    "reports": {
                        "MZ": None,
                        "RT": None,
                        "Height": None,
                        "FWHM": None,
                        "EIC": {
                "rt": [],
                "intensity": []}
                    }
                }
                output_json['metrics'].append(metric)

with open('mzml_summary.json', 'w') as file:
    json.dump(output_json, file, indent=4)
