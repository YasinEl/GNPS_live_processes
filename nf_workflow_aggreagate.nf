#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.json_directory = "/home/yasin/yasin/projects/GNPS_live_processes/random_data"
TOOL_FOLDER = "$baseDir/bin"


// create plots for RT, height and FWHM
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

process create_RT_overview {
    conda "$TOOL_FOLDER/requirements.yml"
    
    publishDir "./nf_output", mode: 'copy'

    input:
    path all_jsons_table
    val toolFolder

    output:
    path("RT_stability_plot.html"), emit: html

    script:
    """
    python $toolFolder/create_STD_overview_plot.py --all_jsons_table ${all_jsons_table} --plot_type rt_stability 
    """
}


process create_realtive_height_overview {
    conda "$TOOL_FOLDER/requirements.yml"
    
    publishDir "./nf_output", mode: 'copy'

    input:
    path all_jsons_table
    val toolFolder

    output:
    path("Height_plot.html"), emit: html

    script:
    """
    python $toolFolder/create_STD_overview_plot.py --all_jsons_table ${all_jsons_table} --plot_type intensity_stability --normalize True 
    """
}

workflow {
    json_directory = Channel.from(params.json_directory)
    all_jsons_table = aggregateJsonsToTable(json_directory, TOOL_FOLDER)
    all_jsons_json = aggregateJsonsToJson(json_directory, TOOL_FOLDER)
    //create_RT_overview(all_jsons_table, TOOL_FOLDER) //wont work yet because there is no injection column)

}