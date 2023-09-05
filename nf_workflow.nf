#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.mzml_files = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/14494_19NAVY52_BLOOD_15_EDTA_pos_138.mzML"  // Default location; you can specify a different path when you run the script
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

process ApplyFeatureFinderMetabo {
    //conda 'openms'
    conda "bioconda::openms=2.9.1"

    input:
    path mzml_file
    val toolFolder

    output:
    path("${mzml_file}.featureXML"), emit: featureXML

    script:
    """
    FeatureFinderMetabo -in ${mzml_file} -out ${mzml_file}.featureXML -algorithm:epd:width_filtering "auto" -algorithm:ffm:report_convex_hulls true
    """
}

process featureXML2csv {
    //conda 'openms'
    conda "bioconda::openms=2.9.1"

    input:
    path featureXML_file
    val toolFolder

    publishDir "./nf_output", mode: 'copy'

    output:
    path("${featureXML_file}.csv"), emit: csv

    script:
    """
    TextExporter -in ${featureXML_file} -out ${featureXML_file}.csv -feature:add_metavalues 100
    """
}

workflow {
    mzml_files = Channel.fromPath(params.mzml_files)
    ms2_counts = CountMS2Scans(mzml_files, TOOL_FOLDER)
    feature_list = ApplyFeatureFinderMetabo(mzml_files)
    op = featureXML2csv(feature_list)
}

