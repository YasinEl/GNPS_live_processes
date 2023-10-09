#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.json_directory = "/home/yasin/yasin/projects/GNPS_live_processes/random_data"
TOOL_FOLDER = "$baseDir/bin"

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

process aggregateJsonsToJson {
    conda "$TOOL_FOLDER/requirements.yml"
    
    publishDir "./nf_output", mode: 'copy'

    input:
    path json_directory
    val toolFolder

    output:
    path("mzml_summary_aggregation.json"), emit: json

    script:
    """
    python $toolFolder/aggregateJsonsToJson.py --json_directory ${json_directory}
    """
}

process jsontosql {
    conda "$TOOL_FOLDER/requirements.yml"
    
    publishDir "./nf_output", mode: 'copy'

    input:
    path jsonjson_directory
    val toolFolder

    output:
    path("aggregated_summary.db"), emit: json

    script:
    """
    python $toolFolder/jsontosql.py --json_path ${jsonjson_directory}
    """
}

workflow {
    json_directory = Channel.from(params.json_directory)
    all_jsons_table = aggregateJsonsToTable(json_directory, TOOL_FOLDER)
    all_jsons_json = aggregateJsonsToJson(json_directory, TOOL_FOLDER)
    jsontosql(all_jsons_json, TOOL_FOLDER)
}