#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.mzml_files = ""  // Default location; you can specify a different path when you run the script
TOOL_FOLDER = "$baseDir/bin"


process CountMS2Scans {
    conda "$TOOL_FOLDER/requirements.yml"
    
    publishDir "./nf_output", mode: 'copy'

    input:
    path mzml_file
    val toolFolder

    output:
    path("${mzml_file}.json"), emit: json

    script:
    """
    python $toolFolder/count_ms2.py $mzml_file > ${mzml_file}.json
    """
}


workflow {
    mzml_files = Channel.fromPath(params.mzml_files)
    ms2_counts = CountMS2Scans(mzml_files, TOOL_FOLDER)
}

