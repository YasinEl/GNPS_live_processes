#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.mzml_files = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/Pool.mzML"
params.parameter_file = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/parameter file.xlsx"  // Default location; you can specify a different path when you run the script
TOOL_FOLDER = "$baseDir/bin"


process CountMS2Scans {
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

process Prepare_json_for_output_collection {
    conda "$TOOL_FOLDER/requirements.yml"
    
    publishDir "./nf_output", mode: 'copy'

    input:
    path mzml_file
    val toolFolder

    output:
    path("mzml_summary.json"), emit: json

    script:
    """
    python $toolFolder/prepare_json_for_output_collection.py $mzml_file > mzml_summary.json
    """
}

process ApplyFeatureFinderMetabo {
    //conda 'openms'
    conda "bioconda::openms=2.9.1"

    input:
    path mzml_file

    output:
    path("features.featureXML"), emit: featureXML

    script:
    """
    FeatureFinderMetabo -in ${mzml_file} -out features.featureXML -algorithm:epd:width_filtering "auto" -algorithm:ffm:report_convex_hulls true -algorithm:mtd:mass_error_ppm 10 -algorithm:common:noise_threshold_int 10000 -algorithm:ffm:remove_single_traces true
    """
}

process featureXML2csv {
    //conda 'openms'
    conda "bioconda::openms=2.9.1"

    input:
    path featureXML_file

    publishDir "./nf_output", mode: 'copy'

    output:
    path("features_adducts.csv"), emit: csv

    script:
    """
    TextExporter -in ${featureXML_file} -out features_adducts.csv -feature:add_metavalues 100
    """
}

process ApplyMetaboliteAdductDecharger {
    //conda 'openms'
    conda "bioconda::openms=2.9.1"

    input:
    path featureXML_file
    val toolFolder

    publishDir "./nf_output", mode: 'copy'

    output:
    path("feature_adducts.featureXML"), emit: featureXML

    script:
    """
    MetaboliteAdductDecharger -in ${featureXML_file} -out_fm feature_adducts.featureXML 
    """
}

workflow {
    mzml_files = Channel.fromPath(params.mzml_files)
    //ms2_counts = CountMS2Scans(mzml_files, TOOL_FOLDER)
    output_json = Prepare_json_for_output_collection(mzml_files, TOOL_FOLDER)
    feature_list = ApplyFeatureFinderMetabo(mzml_files)
    feature_list_w_adducts = ApplyMetaboliteAdductDecharger(feature_list, TOOL_FOLDER)
    feature_list_csv = featureXML2csv(feature_list_w_adducts)
}

