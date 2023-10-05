#!/bin/bash


#can be run with ./test_many.sh /home/yasin/yasin/projects/GNPS_live_processes/big_test_set 

# Directory containing mzML files
input_dir=$1

# Output directory for Nextflow
output_dir="/home/yasin/yasin/projects/GNPS_live_processes/nf_output"

# Loop through all mzML files in the input directory
for mzml_file in "$input_dir"/*.mzML; do
    # Run the Nextflow script on the current mzML file
    nextflow run nf_workflow.nf --mzml_files "$mzml_file" --parameter_file /home/yasin/yasin/projects/GNPS_live_processes/data/demo_parameters.xlsx

    # Find a unique number for renaming the output JSON file
    counter=0
    while [ -e "$output_dir/mzml_summary_${counter}.json" ]; do
        ((counter++))
    done

    # Rename the output file
    mv "$output_dir/mzml_summary.json" "$output_dir/mzml_summary_${counter}.json"
done
