#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.mzml_files = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/sixmix.mzML"
params.parameter_file = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/parameter_file.xlsx" 
TOOL_FOLDER = "$baseDir/bin"


// take folder with all jsons and aggreagte then into a single table
// make a different table for each set or make groupable by set
// 


process aggregate_jsons {
    conda "$TOOL_FOLDER/requirements.yml"
    
    publishDir "./nf_output", mode: 'copy'

    input:
    path mzml_file
    val toolFolder

    output:
    path("mzml_summary.json"), emit: json

    script:
    """
    python $toolFolder/count_ms2.py $mzml_file > mzml_summary.json
    """
}
