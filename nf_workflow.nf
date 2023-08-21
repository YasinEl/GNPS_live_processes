#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.mzml_files = ""  // Default location; you can specify a different path when you run the script
TOOL_FOLDER = "$baseDir"


process CountMS2Scans {
    conda "$TOOL_FOLDER/requirements.yml"
    
    input:
    path mzml_file
    val toolFolder

    output:
    path("${mzml_file}.count"), emit: count

    script:
    """
    python $toolFolder/count_ms2.py $mzml_file > ${mzml_file}.count
    """
}


workflow {
    mzml_files = Channel.fromPath("${params.mzml_files}/*.mzML")
    ms2_counts = CountMS2Scans(mzml_files, TOOL_FOLDER)
}

