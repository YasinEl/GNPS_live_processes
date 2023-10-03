from xml.etree import ElementTree as ET

# Initialize variables to store sum and count for averaging
sum_lower_limit = 0
sum_upper_limit = 0
count_lower_limit = 0
count_upper_limit = 0

# Load mzML file
tree = ET.parse('/home/yasin/yasin/projects/GNPS_live_processes/random_data/agilent.mzML')
root = tree.getroot()

# Define namespace to navigate XML tree
namespaces = {'ns': 'http://psi.hupo.org/ms/mzml'}

# Iterate through each spectrum
for spectrum in root.findall(".//ns:spectrum", namespaces=namespaces):
    ms_level = spectrum.find(".//ns:cvParam[@accession='MS:1000511']", namespaces=namespaces)
    
    if ms_level is not None and ms_level.attrib['value'] == "1":
        lower_limit = spectrum.find(".//ns:cvParam[@accession='MS:1000501']", namespaces=namespaces)
        upper_limit = spectrum.find(".//ns:cvParam[@accession='MS:1000500']", namespaces=namespaces)
        
        if lower_limit is not None:
            sum_lower_limit += float(lower_limit.attrib['value'])
            count_lower_limit += 1
        
        if upper_limit is not None:
            sum_upper_limit += float(upper_limit.attrib['value'])
            count_upper_limit += 1

# Calculate average
if count_lower_limit > 0:
    avg_lower_limit = sum_lower_limit / count_lower_limit
else:
    avg_lower_limit = 'NA'

if count_upper_limit > 0:
    avg_upper_limit = sum_upper_limit / count_upper_limit
else:
    avg_upper_limit = 'NA'

print(f'Average lower limit: {avg_lower_limit}')
print(f'Average upper limit: {avg_upper_limit}')
