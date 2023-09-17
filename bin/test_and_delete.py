




import pyopenms as oms

def calculate_tic(file_path):
    # Initialize the MSExperiment instance
    exp = oms.MSExperiment()
    oms.MzMLFile().load(file_path, exp)
    
    # Calculate TIC for each spectrum and store in a list
    tic_values = [spectrum.calculateTIC() for spectrum in exp]

    return tic_values



mzml_path = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/sixmix.mzML"

tics = calculate_tic(mzml_path)
print(tics)