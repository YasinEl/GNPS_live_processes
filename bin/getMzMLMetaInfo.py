from pyteomics import mzml, auxiliary





if __name__ == "__main__":
    
    mzml_path = '/home/yasin/yasin/projects/GNPS_live_processes/random_data/sixmix.mzML'

    import pymzml

    run = pymzml.run.Reader(mzml_path)

    # Accessing run information
    run_info = run.info

    # Attempting to fetch the timestamp
    start_time = run_info.get('startTimeStamp')

    if start_time:
        print(f"Start timestamp: {start_time}")
    else:
        print("Timestamp not found.")