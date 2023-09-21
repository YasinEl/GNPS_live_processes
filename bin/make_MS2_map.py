import pandas as pd
import argparse
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# Function to categorize MS2 features based on different thresholds
def assign_MS2_groups(df, peak_count_threshold, purity_threshold, intensity_ratio_threshold, collision_energy_ratio):
    
    # Identify the highest precursor intensity for each group
    df['highest_prec_int'] = df.groupby(['Group', 'Associated Feature Label', 'Collision energy'])['Precursor intensity'].transform(max) == df['Precursor intensity']
    
    # Assign 'no MS2' to rows where MS Level is NaN
    df.loc[df['MS Level'].isna(), 'f_MS2'] = 'no MS2'

    # The following lines categorize the MS2 features based on multiple criteria such as peak count, purity, and intensity.

    # Good quality MS2 categorization
    df.loc[(df['Peak count (filtered)'] >= peak_count_threshold) & (df['Purity'] > purity_threshold) & (df['Feature Apex intensity'] > 0), 'f_MS2'] = f'≥{str(peak_count_threshold)} fragments'
    
    # Bad quality due to apex triggering
    df.loc[(df['Peak count (filtered)'] < peak_count_threshold) & (df['Prec-Apex intensity ratio'] < intensity_ratio_threshold) & (df['Feature Apex intensity'] > 0), 'f_MS2'] = f'<{str(peak_count_threshold)} fragments; triggered {str(intensity_ratio_threshold * 100)}% of apex'
    
    # Bad quality due to impurities
    df.loc[(df['Peak count (filtered)'] >= peak_count_threshold) & (df['Purity'] <= purity_threshold) & (df['Feature Apex intensity'] > 0), 'f_MS2'] = f'≥{str(peak_count_threshold)} fragments; precursor purity <{(purity_threshold)*100}%'
    
    # Bad quality due to higher collision energy
    df.loc[(df['Peak count (filtered)'] <= peak_count_threshold) & ((df['Precursor intensity in MS2'] / df['Max fragment intensity']) > collision_energy_ratio) & (df['Feature Apex intensity'] > 0), 'f_MS2'] = f'<{str(peak_count_threshold)} fragments; precursor ion {str(collision_energy_ratio * 100)}% of MS2 base peak'
    
    # Mark vacant MS2 spots
    df.loc[df['Feature Apex intensity'] == 0, 'f_MS2'] = 'vacant MS2'
    
    # General bad MS2 categorization
    df.loc[(df['Peak count (filtered)'] < peak_count_threshold) & (df['f_MS2'].isna()) & (df['Feature Apex intensity'] > 0), 'f_MS2'] = f'<{str(peak_count_threshold)} fragments'
    
    # Mark redundant MS2 features
    df.loc[(df['highest_prec_int'] == False) & (df['Feature Apex intensity'] > 0) & ~df['MS Level'].isna(), 'f_MS2'] = 'redundant MS2'
    
    # Fill NaN or zero apex intensities with precursor intensities
    df.loc[df['Feature Apex intensity'].isna() | (df['Feature Apex intensity'] == 0), 'Feature Apex intensity'] = df['Precursor intensity']

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MS2 data.')
    parser.add_argument('--file_path', type=str, help='Path to the CSV file')
    parser.add_argument('--peak_count_threshold', type=int, default=6, help='Threshold for peak count')
    parser.add_argument('--purity_threshold', type=float, default=0.7, help='Threshold for purity')
    parser.add_argument('--intensity_ratio_threshold', type=float, default=0.3, help='Threshold for intensity ratio')
    parser.add_argument('--collision_energy_ratio', type=float, default=0.6, help='Threshold for collision energy ratio')
    
    args = parser.parse_args()

    df = pd.read_csv(args.file_path, dtype={
        "MS Level": float,
        "Precursor charge": float,
        "Max fragment intensity": float,
        "Precursor intensity": float,
        "Purity": float,
        "Peak count": float,
        "Peak count (filtered)": float,
        "Feature Apex intensity": float,
        "Rel Feature Apex distance": float,
        "Prec-Apex intensity ratio": float,
        "Precursor intensity in MS2": float
    })



    df = assign_MS2_groups(df, args.peak_count_threshold, args.purity_threshold, args.intensity_ratio_threshold, args.collision_energy_ratio)

    # Create a color map dictionary to reuse for both fill and borders
    color_map = {
        'vacant MS2': '#E41A1C',
        'no MS2': '#377EB8',
        f'≥{str(args.peak_count_threshold)} fragments': '#4DAF4A',
        'redundant MS2': '#984EA3',
        f'<{str(args.peak_count_threshold)} fragments; precursor purity <{(args.purity_threshold)*100}%': '#FF7F00',
        f'<{str(args.peak_count_threshold)} fragments; precursor ion {str(args.collision_energy_ratio * 100)}% of MS2 base peak': '#000000',
        f'<{str(args.peak_count_threshold)} fragments': '#A65628',
        f'<{str(args.peak_count_threshold)} fragments; triggered {str(args.intensity_ratio_threshold * 100)}% of apex': '#F781BF'
    }

    filtered_color_map = {k: v for k, v in color_map.items() if k in df['f_MS2'].unique()}


    # Initialize an empty figure
    fig = go.Figure()

    # Loop through each category in f_MS2 to plot each as its own scatter series
    for category in df['f_MS2'].unique():
        sub_df = df[df['f_MS2'] == category]
        color = filtered_color_map.get(category, '#808080')
        
        fig.add_trace(go.Scatter(
            x=sub_df['Retention Time (min)'],
            y=sub_df['Precursor m/z'],
            mode='markers',
            name=category,
            marker=dict(
                color=color,
                line=dict(color=color, width=2),
                size=6,
                opacity=0.7
            )
        ))

    fig.write_html("MS2map.html")


    
    fig.write_html("MS2map.html")

