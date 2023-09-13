#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.mzml_files = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/sixmix.mzML"
params.parameter_file = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/parameter_file.xlsx"  // Default location; you can specify a different path when you run the script
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


process PrepareForFeatureFinderMetaboIdent {
    conda "$TOOL_FOLDER/requirements.yml"

    input:
    path prepared_parameters
    val toolFolder

    output:
    path("*.tsv"), emit: tsv

    script:
    """
    python $toolFolder/PrepareForFeatureFinderMetaboIdent.py --parameter_json ${prepared_parameters}
    """
}

process HandleParameterFile {
    conda "$TOOL_FOLDER/requirements.yml"

    input:
    path parameter_file
    val mzml_file
    val toolFolder

    output:
    path("prepared_parameters.json"), emit: json
    
    script:
    """
    python $toolFolder/handle_parameter_file.py --file_path ${parameter_file} --mzml_path ${mzml_file}
    """
}

process ApplyFeatureFinderMetaboIdent {
    //conda 'openms'
    conda "bioconda::openms=2.9.1"

    input:
    path mzml_file
    each standard_set

    output:
    path("*.featureXML"), emit: featureXML, optional: true

    script:
    """
    FeatureFinderMetaboIdent -in ${mzml_file} -id ${standard_set} -out ${standard_set.baseName}.featureXML -extract:n_isotopes 4 
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

process featureXML_targeted2csv {
    //conda 'openms'
    conda "bioconda::openms=2.9.1"

    input:
    path featureXML_file

    publishDir "./nf_output", mode: 'copy'

    output:
    path("*.csv"), emit: csv

    script:
    """
    TextExporter -in ${featureXML_file} -out ${featureXML_file.baseName}.csv -feature:add_metavalues 100
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


process Add_targeted_standard_extracts_to_output_collection {
    conda "$TOOL_FOLDER/requirements.yml"
    
    publishDir "./nf_output", mode: 'copy'

    input:
    path targeted_feature_list_csv
    path output_json
    val toolFolder

    output:
    path("mzml_summary.json"), emit: json

    script:
    """
    python $toolFolder/add_targeted_info_to_output_json.py --output_json_path ${output_json} --std_sets ${targeted_feature_list_csv.join(",")}
    """
}

workflow {

    //setup parameters and workflow structure
    mzml_files = Channel.from(params.mzml_files)
    parameter_file = Channel.from(params.parameter_file)
    prepared_parameters = HandleParameterFile(parameter_file, mzml_files, TOOL_FOLDER)
    output_json = Prepare_json_for_output_collection(params.mzml_files, TOOL_FOLDER)

    //targeted standard extraction
    PrepareForFeatureFinderMetaboIdent(prepared_parameters, TOOL_FOLDER)
    openms_std_output = ApplyFeatureFinderMetaboIdent(mzml_files, PrepareForFeatureFinderMetaboIdent.out.tsv.collect())
    output_json_targeted = Add_targeted_standard_extracts_to_output_collection(ApplyFeatureFinderMetaboIdent.out.featureXML.collect(), output_json, TOOL_FOLDER)


    //feature_list = ApplyFeatureFinderMetabo(params.mzml_files)
    //feature_list_w_adducts = ApplyMetaboliteAdductDecharger(feature_list, TOOL_FOLDER)
    //feature_list_csv = featureXML2csv(feature_list_w_adducts)
    
}

