
from pyopenms import MSExperiment, MzMLFile

# Load the mzML file into an MSExperiment object
filename = "/home/yasin/yasin/projects/GNPS_live_processes/random_data/sixmix.mzML"
exp = MSExperiment()
MzMLFile().load(filename, exp)

# Loop through spectra to find MS2 scans
for spectrum in exp:
    if spectrum.getMSLevel() == 2:
        precursors = spectrum.getPrecursors()
        collision_energy = precursors[0].getMetaValue(b'collision energy')
        print(f"Collision Energy: {collision_energy}")
