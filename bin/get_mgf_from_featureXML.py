import pyopenms as oms

def get_MS2_for_features(feature_file, mzml_file, mz_tolerance_ppm=10):

    output_mgf = 'output.mgf'
    # Load featureXML and mzML
    feature_map = oms.FeatureMap()
    oms.FeatureXMLFile().load(feature_file, feature_map)

    ms_exp = oms.MSExperiment()
    oms.MzMLFile().load(mzml_file, ms_exp)

    relevant_spectra = []

    for feature in feature_map:
        convex_hull = feature.getConvexHulls()[0]  # Assuming one convex hull per feature
        points = convex_hull.getHullPoints()
        
        rt_values = [point[0] for point in points]
        rt_start = min(rt_values)
        rt_end = max(rt_values)

        mz_feature = feature.getMZ()
        mz_tol = (mz_tolerance_ppm * mz_feature) / 1e6
        
        apex_intensity = feature.getIntensity()

        # Extract all MS2 spectra that fall within the RT boundaries and m/z tolerance of the feature
        for spectrum in ms_exp:
            if spectrum.getMSLevel() == 2:
                mz_spectrum = spectrum.getPrecursors()[0].getMZ()
                if rt_start <= spectrum.getRT() <= rt_end and (mz_feature - mz_tol) <= mz_spectrum <= (mz_feature + mz_tol):
                    # Get the precursor intensity and add it to the title.
                    precursor_intensity = spectrum.getPrecursors()[0].getIntensity()
                    title = f"scan={spectrum.getNativeID()}_PrecursorIntensity={precursor_intensity}_ApexIntensity={apex_intensity}"
                    spectrum.setMetaValue("TITLE", title)
                    relevant_spectra.append(spectrum)

    # Save to a single MGF file
    exp_to_store = oms.MSExperiment()
    for spec in relevant_spectra:
        exp_to_store.addSpectrum(spec)
    oms.MascotGenericFile().store(output_mgf, exp_to_store)

featureXML_path = "work/56/a5d3861348435b27104724e612fddd/feature_adducts.featureXML"
mzML_path = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/sixmix.mzML"
get_MS2_for_features(featureXML_path, mzML_path)
