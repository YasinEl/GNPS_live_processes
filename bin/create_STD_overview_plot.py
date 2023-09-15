import pandas as pd
import plotly.express as px
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Create a plotly html from all_jsons_table.")
    parser.add_argument('--all_jsons_table', type=str, help="Path to all_jsons_table.")
    parser.add_argument('--plot_type', type=str, help="Type of plot to be created.")
    parser.add_argument('--normalize', type=str2bool, help="Shall plots be normalized to 100 (for height).", default=False)

    
    args = parser.parse_args()

    all_jsons_table = args.all_jsons_table
    plot_type = args.plot_type
    normalize = args.normalize

    #all_jsons_table = "/home/yasin/yasin/projects/GNPS_live_processes/nf_output/all_jsons_table.csv"
    #plot_type = 'intensity_stability'
    
    df = pd.read_csv(all_jsons_table)


    if plot_type == 'rt_stability':
        # Compute the median RT for each 'name'
        medians = df.groupby('name')['RT'].median().rename('median_RT')

        # Merge the median values back to the original dataframe
        df = df.merge(medians, on='name', how='left')

        # Compute the difference from the median for each entry
        df['RT_diff_from_median'] = df['RT'] - df['median_RT']

        label_changes = {
            'injection': 'injection number',
            'RT_diff_from_median': 'Difference from Median RT [s]',
            'name': 'molecule_adduct'
        }

        # Create the figure
        fig = px.line(df, x='injection', y='RT_diff_from_median', color='name', labels=label_changes, hover_data=['mzml_name'])

        # Save as HTML
        fig.write_html('RT_stability_plot.html')


    if plot_type == 'intensity_stability':
        y_label = "Height"

        if normalize == True:
            # Normalize the Height column to 100 for each compound
            df['Height'] = df.groupby('name')['Height'].transform(lambda x: (x / x.max()) * 100)
            y_label = "Normalized Height"

        label_changes = {
            'injection': 'injection number',
            'Height': y_label,
            'name': 'molecule_adduct'
        }

        # Create the figure
        fig = px.line(df, x='injection', y='Height', color='name', labels=label_changes, hover_data=['mzml_name'])

        # Save as HTML
        fig.write_html('Height_plot.html')

