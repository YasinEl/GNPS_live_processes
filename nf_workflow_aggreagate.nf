#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.json_directory = "/home/yasin/yasin/projects/GNPS_live_processes/nf_output"
params.parameter_file = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/parameter_file.xlsx" 
TOOL_FOLDER = "$baseDir/bin"


// take folder with all jsons and aggreagte then into a single table OR
// make a different table for each set or make groupable by set (for STD plots)
// 


process aggregateJsonsToTable {
    conda "$TOOL_FOLDER/requirements.yml"
    
    publishDir "./nf_output", mode: 'copy'

    input:
    path json_directory
    val toolFolder

    output:
    path("all_jsons_table.csv"), emit: csv

    script:
    """
    python $toolFolder/aggregateJsonsToStandardTable.py --json_directory ${json_directory}
    """
}


workflow {

    json_directory = Channel.from(params.json_directory)
    all_jsons_table = aggregateJsonsToTable(json_directory, TOOL_FOLDER)



}