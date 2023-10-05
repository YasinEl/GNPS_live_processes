#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.mzml_files = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/onr2.mzML"
// params.mzml_files = ""
params.parameter_file = "$workflow.projectDir/data/demo_parameters.xlsx"
params.timeOfUpload = '2023-09-21 13:45:30'
TOOL_FOLDER = "$baseDir/bin"


//need to screen M+x isotopes for MS2 mapping
//take care of pos/neg switching
//take care of no MS2 spectra
//corrupted raw/mzml files
//take care of different mass ranges

process CreateMS2Inventory {
    conda "$TOOL_FOLDER/requirements.yml"
    //conda "bioconda::openms=2.9.1"

    input:
    path mzml_file
    path featureXML
    val general_parameters
    val toolFolder

    output:
    path("MS2_inventory_table.csv"), emit: csv

    script:
    """
    ms2ppm=\$(echo "${general_parameters}" | cut -d',' -f2)
    python $toolFolder/createMS2table.py --file_path ${mzml_file} --featureXML_path ${featureXML} --ppm \$ms2ppm
    """
}


process Prepare_json_for_output_collection {
    conda "$TOOL_FOLDER/requirements.yml"
    
    //publishDir "./nf_output", mode: 'copy'

    input:
    path mzml_file
    val toolFolder

    output:
    path("mzml_summary.json"), emit: json

    script:
    """
    python $toolFolder/prepare_json_for_output_collection.py --filename ${mzml_file} --datetime '${params.timeOfUpload}'
    """
}

process ApplyFeatureFinderMetabo {
    //conda 'openms'
    //conda "bioconda::openms=2.9.1"
    conda "$TOOL_FOLDER/requirements.yml"

    input:
    path mzml_file
    val general_parameters

    output:
    path("features.featureXML"), emit: featureXML

    script:
    """
    firstParam=\$(echo "${general_parameters}" | cut -d',' -f1)

    FeatureFinderMetabo -in ${mzml_file} -out features.featureXML -algorithm:epd:width_filtering off -threads 10 \
    -algorithm:ffm:report_convex_hulls true  -algorithm:ffm:mz_scoring_by_elements true -algorithm:ffm:elements CHNOPSClNaKFBrLiMgSiCaCrFeCuSe \
    -algorithm:common:chrom_fwhm 2 -algorithm:ffm:use_smoothed_intensities false -algorithm:mtd:mass_error_ppm \$firstParam \
    -algorithm:common:noise_threshold_int 1000 -algorithm:ffm:remove_single_traces true -algorithm:mtd:quant_method max_height \
    -algorithm:mtd:min_trace_length 2 -algorithm:ffm:charge_upper_bound 20 -algorithm:ffm:local_rt_range 1 -algorithm:ffm:local_mz_range 15
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
    conda "$TOOL_FOLDER/requirements.yml"

    input:
    path mzml_file
    each standard_set
    val general_parameters

    output:
    path("*.featureXML"), emit: featureXML, optional: true

    script:
    """
    if [[ "${standard_set.baseName}" != "set_none" ]]; then
        ms1ppm=\$(echo "${general_parameters}" | cut -d',' -f1)
        FeatureFinderMetaboIdent -in ${mzml_file} -id ${standard_set} -out ${standard_set.baseName}.featureXML -threads 5 -extract:n_isotopes 2 -extract:mz_window \$ms1ppm
    else
        cp ${standard_set} set_none.featureXML
    fi
    """
}

process featureXML2csv {
    //conda 'openms'
    //conda "bioconda::openms=2.9.1"
    conda "$TOOL_FOLDER/requirements.yml"

    input:
    path featureXML_file

    output:
    path("features_adducts.csv"), emit: csv

    script:
    """
    TextExporter -in ${featureXML_file} -out features_adducts.csv -feature:add_metavalues 0
    """
}

process featureXML_targeted2csv {
    //conda 'openms'
    // conda "bioconda::openms=2.9.1"
    conda "$TOOL_FOLDER/requirements.yml"

    input:
    path featureXML_file

    output:
    path("*.csv"), emit: csv

    script:
    """
    TextExporter -in ${featureXML_file} -out ${featureXML_file.baseName}.csv -feature:add_metavalues 0
    """
}


process ApplyMetaboliteAdductDecharger {
    //conda 'openms'
    // conda "bioconda::openms=2.9.1"
    conda "$TOOL_FOLDER/requirements.yml"

    input:
    path featureXML_file
    val toolFolder

    output:
    path("feature_adducts.featureXML"), emit: featureXML

    script:
    """
    MetaboliteAdductDecharger -in ${featureXML_file} -out_fm feature_adducts.featureXML -threads 5 -algorithm:MetaboliteFeatureDeconvolution:max_neutrals 3
    """
}


process Add_targeted_standard_extracts_to_output_collection {
    conda "$TOOL_FOLDER/requirements.yml"

    input:
    path targeted_feature_list_featurexml
    path targeted_feature_list_tsv
    path output_json
    val toolFolder

    output:
    path("mzml_summary.json"), emit: json

    script:
    """
    python $toolFolder/add_targeted_info_to_output_json.py --output_json_path ${output_json} --std_sets ${targeted_feature_list_featurexml.join(",")} \
    --std_sets_tsvs ${targeted_feature_list_tsv.join(",")}
    """
}

process Add_MS2_info_to_output_collection {
    conda "$TOOL_FOLDER/requirements.yml"
    
    //publishDir "./nf_output", mode: 'copy'

    input:
    path output_json
    path MS2_table
    val toolFolder

    output:
    path("mzml_summary.json"), emit: json

    script:
    """
    python $toolFolder/add_MS2_info_to_output_json.py --output_json_path ${output_json} --ms2_inv_csv ${MS2_table} 
    """
}

process CreateMS1Inventory {
    conda "$TOOL_FOLDER/requirements.yml"
    //conda "bioconda::openms=2.9.1"

    input:
    path mzml_file
    val toolFolder

    output:
    path("MS1_inventory_table.csv"), emit: csv

    script:
    """
    python $toolFolder/createMS1table.py --file_path ${mzml_file} 
    """
}


process Add_MS1_info_to_output_collection {
    conda "$TOOL_FOLDER/requirements.yml"
    
    publishDir "./nf_output", mode: 'copy'

    input:
    path output_json
    path MS1_table
    path feature_table
    val toolFolder

    output:
    path("mzml_summary.json"), emit: json

    script:
    """
    python $toolFolder/add_MS1_info_to_output_json.py --output_json_path ${output_json} --ms1_inv_csv ${MS1_table} --feature_csv ${feature_table}
    """
}

process CreateMS2map {
    conda "$TOOL_FOLDER/requirements.yml"
    
    publishDir "./nf_output", mode: 'copy'

    input:
    path ms2inventory_csv
    val toolFolder

    output:
    path("MS2map.html"), emit: json

    script:
    """
    python $toolFolder/make_MS2_map.py --file_path ${ms2inventory_csv}
    """
}

process getParametersFromJson {
    conda "$TOOL_FOLDER/requirements.yml"

    input:
    path ParamJsonPath

    output:
    stdout 

    script:
    """
    #!/usr/bin/env python
    import json
    import pandas as pd
    from io import StringIO

    with open('$ParamJsonPath', 'r') as f:
        jsonObject = json.load(f)
        jsonObject = {key: pd.read_json(StringIO(value)) if isinstance(value, str) else value for key, value in jsonObject.items() if value}

    ms1_precision = jsonObject['MS1 precision']
    ms2_precision = jsonObject['MS2 precision']

    print(ms1_precision, ms2_precision, sep=',', end='')
    """
}

workflow {
    //setup parameters and workflow structure
    mzml_files_ch = Channel.from(params.mzml_files)
    parameter_file_ch = Channel.from(params.parameter_file)

    prepared_parameters = HandleParameterFile(parameter_file_ch, mzml_files_ch, TOOL_FOLDER)
    paramList = getParametersFromJson(prepared_parameters)
    output_json = Prepare_json_for_output_collection(mzml_files_ch, TOOL_FOLDER) 

    //targeted standard extraction
    PrepareForFeatureFinderMetaboIdent(prepared_parameters, TOOL_FOLDER)
    openms_std_output = ApplyFeatureFinderMetaboIdent(mzml_files_ch, PrepareForFeatureFinderMetaboIdent.out.tsv.collect(), paramList)
    output_json_targeted = Add_targeted_standard_extracts_to_output_collection(ApplyFeatureFinderMetaboIdent.out.featureXML.collect(), PrepareForFeatureFinderMetaboIdent.out.tsv.collect(), output_json, TOOL_FOLDER)

    //untargeted feature extraction
    feature_list = ApplyFeatureFinderMetabo(mzml_files_ch, paramList)
    feature_list_w_adducts = ApplyMetaboliteAdductDecharger(feature_list, TOOL_FOLDER)
    feature_list_csv = featureXML2csv(feature_list_w_adducts)
    
    //collect untargeted MS1, MS2, and feature information
    MS2_inventory = CreateMS2Inventory(mzml_files_ch, feature_list_w_adducts, paramList, TOOL_FOLDER)
    output_json_ms2 = Add_MS2_info_to_output_collection(output_json_targeted, MS2_inventory, TOOL_FOLDER)

    MS1_inventory = CreateMS1Inventory(mzml_files_ch, TOOL_FOLDER)
    output_json_ms1 = Add_MS1_info_to_output_collection(output_json_ms2, MS1_inventory, feature_list_csv, TOOL_FOLDER)
}

