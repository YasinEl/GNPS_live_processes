import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
from plotly.colors import find_intermediate_color
import plotly.express as px
from plotly.express.colors import sample_colorscale

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import pandas as pd
import re

def create_filtered_table(json_file, name=None, type_=None, collection=None, include_keys=None):
    with open(json_file, 'r') as f:
        data = json.load(f)

    table_data = []

    for entry in data:
        mzml_name = entry.get('mzml_name', None)
        time_of_upload = entry.get('time_of_upload', None)
        metrics = entry.get('metrics', [])

        for metric in metrics:
            metric_name = metric.get('name', None)
            metric_type = metric.get('type', None)
            metric_collection = metric.get('collection', None)

            if ((name is None or name == metric_name) and
                    (type_ is None or type_ == metric_type) and
                    (collection is None or collection == metric_collection)):

                row = {
                    'mzml_file': mzml_name,
                    'date_time': time_of_upload,
                    'name': metric_name,
                    'type': metric_type,
                    'collection': metric_collection
                }

                reports = metric.get('reports', {})

                for key, value in reports.items():
                    if include_keys is None:
                        if not isinstance(value, (list, dict)):
                            row[key] = value
                    else:
                        if key in include_keys:
                            if isinstance(value, dict):
                                if isinstance(list(value.values())[0], dict):
                                    for key2, value2 in value.items():
                                        row[key2] = list(value2.values())
                                else:
                                    row.update(value)
                            else:
                                row[key] = value

                table_data.append(row)

    df = pd.DataFrame(table_data)

    # Identify columns containing lists
    list_columns = [col for col in df.columns if df[col].apply(isinstance, args=(list,)).any()]

    # Explode the DataFrame based on list columns
    if list_columns:
        df = df.explode(list_columns)
        df.reset_index(drop=True, inplace=True)

    return df


def assign_MS2_groups(df_input, peak_count_threshold, purity_threshold, intensity_ratio_threshold, collision_energy_ratio):
    # Identify the highest precursor intensity for each group
    df = df_input.copy()
    df['highest_prec_int'] = df.groupby(['Group', 'Associated Feature Label', 'Collision energy', 'mzml_file'])[
                                 'Precursor intensity'].transform(max) == df['Precursor intensity']

    # Assign 'no MS2' to rows where MS Level is NaN
    df.loc[df['MS Level'].isna(), 'f_MS2'] = 'no MS2'

    # The following lines categorize the MS2 features based on multiple criteria such as peak count, purity, and intensity.

    # Good quality MS2 categorization
    df.loc[(df['Peak count (filtered)'] >= peak_count_threshold) & (df['Purity'] > purity_threshold) & (
                df['Feature Apex intensity'] > 0), 'f_MS2'] = f'≥{str(peak_count_threshold)} fragments'

    # Bad quality due to apex triggering
    df.loc[(df['Peak count (filtered)'] < peak_count_threshold) & (
                df['Prec-Apex intensity ratio'] < intensity_ratio_threshold) & (df[
                                                                                    'Feature Apex intensity'] > 0), 'f_MS2'] = f'<{str(peak_count_threshold)} fragments; triggered <{str(intensity_ratio_threshold * 100)}% of apex'

    # Bad quality due to impurities
    df.loc[(df['Peak count (filtered)'] >= peak_count_threshold) & (df['Purity'] <= purity_threshold) & (df[
                                                                                                             'Feature Apex intensity'] > 0), 'f_MS2'] = f'≥{str(peak_count_threshold)} fragments; precursor purity <{(purity_threshold) * 100}%'

    # Bad quality due to higher collision energy
    df.loc[(df['Peak count (filtered)'] <= peak_count_threshold) & (
                (df['Precursor intensity in MS2'] / df['Max fragment intensity']) > collision_energy_ratio) & (df[
                                                                                                                   'Feature Apex intensity'] > 0), 'f_MS2'] = f'<{str(peak_count_threshold)} fragments; precursor ion >{str(collision_energy_ratio * 100)}% of MS2 base peak'

    # Mark vacant MS2 spots
    df.loc[df['Feature Apex intensity'] == 0, 'f_MS2'] = 'vacant MS2'

    # General bad MS2 categorization
    df.loc[(df['Peak count (filtered)'] < peak_count_threshold) & (df['f_MS2'].isna()) & (
                df['Feature Apex intensity'] > 0), 'f_MS2'] = f'<{str(peak_count_threshold)} fragments'

    # Mark redundant MS2 features
    df.loc[(df['highest_prec_int'] == False) & (df['Feature Apex intensity'] > 0) & ~df[
        'MS Level'].isna(), 'f_MS2'] = 'redundant MS2'

    # Fill NaN or zero apex intensities with precursor intensities
    df.loc[df['Feature Apex intensity'].isna() | (df['Feature Apex intensity'] == 0), 'Feature Apex intensity'] = df[
        'Precursor intensity']

    return df

def prepare_correlation_analysis(df):
    df['rt_bin'] = (df['rt'] // 3) * 3
    df['rt_bin'] = df['rt_bin'].astype(int)
    df_grouped = df.groupby(['mzml_file', 'rt_bin']).agg({'intensity': 'mean'}).reset_index()

    all_bins = df_grouped['rt_bin'].unique()
    all_files = df_grouped['mzml_file'].unique()

    fill_rows = []
    for f in all_files:
        existing_bins = set(df_grouped[df_grouped['mzml_file'] == f]['rt_bin'])
        missing_bins = set(all_bins) - existing_bins
        for b in missing_bins:
            fill_rows.append([f, b, 0])


    df_fill = pd.DataFrame(fill_rows, columns=['mzml_file', 'rt_bin', 'intensity'])
    df_filled = pd.concat([df_grouped, df_fill], ignore_index=True)

    date_time_mapping = df.groupby('mzml_file')['date_time'].first()
    df_filled['date_time'] = df_filled['mzml_file'].map(date_time_mapping)
    df_filled.sort_values(by='date_time', inplace=True)
    df_filled['datetime_order'] = df_filled['date_time'].rank(method='min').astype(int)

    return df_filled

def prepare_pca_data(df, intensity_column = 'TIC_intensity_complete'):

    df['rt_bin'] = (df['rt'] // 3) * 3
    df['rt_bin'] = df['rt_bin'].astype(int)

    df_grouped = df.groupby(['mzml_file', 'rt_bin']).agg({intensity_column: 'mean'}).reset_index()

    all_bins = df_grouped['rt_bin'].unique()
    all_files = df_grouped['mzml_file'].unique()

    fill_rows = []
    for f in all_files:
        existing_bins = set(df_grouped[df_grouped['mzml_file'] == f]['rt_bin'])
        missing_bins = set(all_bins) - existing_bins
        for b in missing_bins:
            fill_rows.append([f, b, 0])



    df_fill = pd.DataFrame(fill_rows, columns=['mzml_file', 'rt_bin', intensity_column])
    df_filled = pd.concat([df_grouped, df_fill], ignore_index=True)



    X = df_filled.pivot(index='mzml_file', columns='rt_bin', values=intensity_column).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    pca = PCA()
    pca.fit(X_scaled)
    n_pcs = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0][0] + 1

    pca = PCA(n_components=n_pcs)
    principalComponents = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_

    finalDf = pd.DataFrame(principalComponents,
                           columns=[f'PC{i + 1}' for i in range(n_pcs)])
    finalDf['mzml_file'] = X.index

    date_time_mapping = df.groupby('mzml_file')['date_time'].first()
    finalDf['date_time'] = finalDf['mzml_file'].map(date_time_mapping)
    finalDf.sort_values(by='date_time', inplace=True)
    #finalDf['datetime_order'] = np.arange(1, len(finalDf) + 1)
    finalDf['datetime_order'] = finalDf['date_time'].rank(method='min').astype(int)

    order_array = np.array(sorted(finalDf['datetime_order'].unique()))

    correlation_coeffs_order = {}
    for pc in finalDf.columns:
        if "PC" in pc:
            corr_coef = np.corrcoef(order_array, finalDf.groupby('datetime_order')[pc].mean())[0, 1]
            correlation_coeffs_order[pc] = corr_coef
            finalDf.rename(columns={
                pc: f"{pc} ({explained_variance_ratio[int(pc[2:]) - 1]:.2f}; injection order r: {corr_coef:.2f})"},
                           inplace=True)

    top_2_pcs = sorted(correlation_coeffs_order, key=lambda x: abs(correlation_coeffs_order[x]), reverse=True)[:2]

    loadings_df = pd.DataFrame(pca.components_, columns=X.columns,
                               index=[f'PC{i + 1}' for i in range(n_pcs)])


    return finalDf, loadings_df, top_2_pcs


# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])


#load data for everything
df_standards = create_filtered_table('C:/Users/elabi/Downloads/mzml_summary_aggregation.json',  type_="standards")
lcms_df = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  type_="standards", include_keys = 'EIC')
ms1_inv = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  collection="MS1_inventory")
ms1_tic = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  collection="MS1_inventory", include_keys = 'MS1_inventory')
ms2_inv = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  collection="MS2_inventory")
ms2_scans = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  collection="MS2_inventory", include_keys = 'MS2_inventory')
ms1_tic_bins = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  collection="MS1_inventory", include_keys = 'TIC_bins')

#get the lists of options for dropdowns etc
mzml_list = ms1_inv['mzml_file'].unique().tolist()
df_sorted = ms1_inv.sort_values('date_time')
mzml_list_sorted = df_sorted['mzml_file'].unique().tolist()

ms1_tic_bins = ms1_tic_bins[ms1_tic_bins['mzml_file'] == mzml_list[0]]

ms1_tic_bins = ms1_tic_bins[['TIC_bins']]
ms1_tic_bins['bin_id'] = range(1, len(ms1_tic_bins) + 1)

# Create sorted and labeled options
sorted_options = [{'label': f"{i + 1}: {filename}", 'value': filename} for i, filename in enumerate(mzml_list_sorted)]



ms_scan_variables = ['Retention Time (min)', 'Precursor m/z', 'Collision energy',
       'Precursor charge', 'Max fragment intensity', 'Precursor intensity',
       'Purity', 'Peak count', 'Peak count (filtered)',
       'Associated Feature Label', 'Feature Apex intensity', 'FWHM',
       'Rel Feature Apex distance', 'Prec-Apex intensity ratio',
       'Precursor intensity in MS2', 'Apex/Precursor intensity']

# Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Page 1", href="/")),
    ],
    brand="GNPS-live",
    color="dark",
    dark=True,
)

# Main layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Uploaded files"),
                dbc.CardBody([
                    html.Div([
                        dcc.Input(id='text-field', type='text', style={'width': '100%'}),
                    ]),
                    html.Div([
                        html.Button('Select', id='select-mzmls-button', style={'display': 'inline-block'}),
                        html.Button('Unselect', id='unselect-mzmls-button', style={'display': 'inline-block'}),
                    ], style={'width': '100%'}),
                    html.Div([
                        dcc.Checklist(id='mzml-checklist',
                                      options=sorted_options,#[{'label': i, 'value': i} for i in mzml_list],
                                      value=mzml_list,
                                      style={'overflow': 'auto', 'height': '800px', 'white-space': 'nowrap'}),
                    ], style={'width': '100%'}),
                ])
            ])
            ,
            width=2),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Standards', tab_id='standards'),
                dbc.Tab(label='MS2', tab_id='ms2'),
                dbc.Tab(label='MS1', tab_id='ms1'),
            ], id='tabs', active_tab='standards'),
            html.Div(id='tabs-content'),
        ], width=10),
    ]),
    dcc.Store(id='store-df'),
    dcc.Store(id='pca-df'),
    dcc.Store(id='intermediate-value'),
])


@app.callback(
    Output('intermediate-value', 'data'),  # You can use a hidden Div to store intermediate values
    Input('mzml-checklist', 'value')
)
def start_lazy_callbacks(selected_mzml):
    # Perform some operation
    if selected_mzml is not None:
        processed_value = ",".join(selected_mzml)  # Just an example operation
        return processed_value
    return None


@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'active_tab')]
)
def update_tab_content(active_tab):
    return render_tab(active_tab)


@app.callback(
    Output('lcms-plot', 'figure'),
    Input('mzml-checklist', 'value'),
    Input('molecule-dropdown', 'value')
)


def update_lcms_plot(selected_mzml, selected_molecule):
    filtered_lcms_df = lcms_df[lcms_df['mzml_file'].isin(selected_mzml) & lcms_df['name'].eq(selected_molecule)].copy()
    filtered_lcms_df['date_time'] = pd.to_datetime(filtered_lcms_df['date_time'])

    unique_dates = filtered_lcms_df['date_time'].unique()
    unique_dates.sort()
    date_order = {date: i + 1 for i, date in enumerate(unique_dates)}

    filtered_lcms_df['order'] = filtered_lcms_df['date_time'].map(date_order)

    color_map = plt.cm.viridis

    lcms_fig = go.Figure()

    for order in sorted(filtered_lcms_df['order'].unique()):
        single_order_df = filtered_lcms_df[filtered_lcms_df['order'] == order]

        # Normalize 'order' to [0, 1] range for colormap
        normalized_order = (order - 1) / (len(unique_dates) - 1)
        color_value = [int(x * 255) for x in color_map(normalized_order)[:3]]

        color_str = f'rgb({color_value[0]}, {color_value[1]}, {color_value[2]})'

        lcms_fig.add_trace(
            go.Scatter(x=single_order_df['rt'], y=single_order_df['intensity'],
                       mode='lines',
                       line=dict(
                           color=color_str,
                       ),
                       name=f"Injection {order}",
                       customdata=single_order_df['date_time'].dt.strftime("%m-%d  %H:%M:%S"),
                       hovertemplate="%{customdata}<br>Intensity: %{y}")
        )

    lcms_fig.update_layout(height=800,
                           xaxis_title='Retention Time',
                           yaxis_title='Intensity',
                           yaxis=dict(exponentformat='E')
                           )

    return lcms_fig


def render_tab(tab_value):
    if tab_value == 'standards':
        return dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Metrics over time", style={'textAlign': 'center', 'padding': '5px',
                                                         'backgroundColor': 'rgba(128, 128, 128, 0.1)',
                                                         'marginBottom': '20px'}),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Label('Standard set', style={'marginBottom': '10px'}),
                                dcc.Dropdown(
                                    id='set-dropdown',
                                    options=[{'label': i, 'value': i} for i in df_standards['collection'].unique()],
                                    value=df_standards['collection'].unique()[0],
                                    style={'width': '90%'}
                                ),
                            ], width=4, className='align-self-end'),

                            dbc.Col([
                                html.Div([], style={'height': '38px'}),  # spacer
                                dcc.Checklist(
                                    id='relative-scale-checkbox',
                                    options=[{'label': 'Relative Scale', 'value': 'relative'}],
                                    value=[]
                                ),
                            ], width=4, className='align-self-end'),
                        ], style={'margin-top': '10px'}),
                    ]),

                    dcc.Graph(id='subplots', style={'height': '800px', 'marginBottom': '10px'})
                ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                          'marginBottom': '10px'}),
            ], width=6),

            dbc.Col([
                html.Div([
                    html.Div("Extracted ion chromatograms", style={'textAlign': 'center', 'padding': '5px',
                                                                   'backgroundColor': 'rgba(128, 128, 128, 0.1)',
                                                                   'marginBottom': '20px'}),
                    html.Label('Standard to display', style={'marginBottom': '10px'}),
                    dcc.Dropdown(
                        id='molecule-dropdown',
                        options=[{'label': i, 'value': i} for i in df_standards['name'].unique()],
                        value=df_standards['name'].unique()[0],
                        style={'width': '200px'}
                    ),

                    dcc.Graph(id='lcms-plot', style={'height': '800px', 'marginBottom': '10px'})
                ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                          'marginBottom': '10px'}),
            ], width=6)
        ], style={'marginTop': '10px'})


    elif tab_value == 'ms2':
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Slider(
                            id='peak-count-slider',
                            min=1,
                            max=20,
                            step=1,
                            value=4,
                            persistence=True
                        ),
                        html.Label("Peak Count Threshold")
                    ]),

                    html.Div([
                        dcc.Slider(
                            id='purity-slider',
                            min=0,
                            max=1,
                            step=0.1,
                            value=0.5,
                            persistence=True
                        ),
                        html.Label("Purity Threshold")
                    ]),

                    html.Div([
                        dcc.Slider(
                            id='intensity-ratio-slider',
                            min=0,
                            max=1,
                            step=0.1,
                            value=0.3,
                            persistence=True
                        ),
                        html.Label("Intensity Ratio Threshold")
                    ]),

                    html.Div([
                        dcc.Slider(
                            id='collision-energy-slider',
                            min=0,
                            max=1,
                            step=0.1,
                            value=0.7,
                            persistence=True
                        ),
                        html.Label("Collision Energy Ratio")
                    ]),

                    html.Div([
                        html.Button('Update Plots', id='update-button', n_clicks=0)
                    ]),
                ], width=2),

                dbc.Col([
                    html.Div([
                        html.Div("MS2 spectra and unique precursor mz counts", style={'textAlign': 'center', 'padding': '5px',
                                                              'backgroundColor': 'rgba(128, 128, 128, 0.1)'}),
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            children=[
                                dcc.Graph(id='ms2-scan-counts', style={'height': '600px', 'marginBottom': '10px'})
                            ])
                    ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                              'marginBottom': '10px'}),
                ], width=10),
            ], style={'marginTop': '10px'}),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("MS2 scan map", style={'textAlign': 'center', 'padding': '5px',
                                                              'backgroundColor': 'rgba(128, 128, 128, 0.1)',
                                                              'marginBottom': '5px'}),
                        dbc.Row([
                            dbc.Col([
                                html.Label('Select mzML file'),
                                dcc.Dropdown(
                                    id='mzml-file-dropdown',
                                    options=[{'label': i, 'value': i} for i in mzml_list],
                                    value=mzml_list[0],
                                    persistence=True)],
                                width=4),
                            dbc.Col([
                                html.Label('Select x-axis variable'),
                                dcc.Dropdown(
                                    id='MS2-map-x-dropdown',
                                    options=[{'label': i, 'value': i} for i in ms_scan_variables],
                                    value='Retention Time (min)',
                                    persistence=True
                                ),
                                html.Div(style={'height': '5px'}),  # For spacing
                                dcc.Checklist(
                                    id='log10-x-checkbox',
                                    options=[{'label': 'Log10', 'value': 'log10'}],
                                    value=[],
                                    persistence=True
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label('Select y-axis variable'),
                                dcc.Dropdown(
                                    id='MS2-map-y-dropdown',
                                    options=[{'label': i, 'value': i} for i in ms_scan_variables],
                                    value='Apex/Precursor intensity',
                                    persistence=True
                                ),
                                html.Div(style={'height': '5px'}),  # For spacing
                                dcc.Checklist(
                                    id='log10-y-checkbox',
                                    options=[{'label': 'Log10', 'value': 'log10'}],
                                    value=['log10'],
                                    persistence=True
                                )
                            ], width=4),
                        ]),
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            children=[
                                dcc.Graph(id='ms2-plot2', style={'height': '800px', 'marginBottom': '10px'})])
                    ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                              'marginBottom': '10px'}),
                ], width=12),
            ])
        ], style={'marginTop': '10px'})
    elif tab_value == 'ms1':
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Total ion chromatogram (TIC)", style={'textAlign': 'center', 'padding': '5px',
                                                          'backgroundColor': 'rgba(128, 128, 128, 0.1)'}),
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            children=[
                                dcc.Graph(id='tic-plot', style={'height': '600px', 'marginBottom': '10px'})])
                    ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                              'marginBottom': '10px'}),
                ], width=12),  # This makes it span the entire width
            ]),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("TIC Statistics", style={'textAlign': 'center', 'padding': '5px',
                                                                   'backgroundColor': 'rgba(128, 128, 128, 0.1)'}),
                        dcc.Checklist(
                            id='relative-to-median-checkbox',
                            options=[
                                {'label': 'Relative to median', 'value': 'show_relative'}
                            ],
                            value=['show_relative']
                        ),
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            children=[
                                dcc.Graph(id='tic-stats_plot', style={'height': '600px', 'marginBottom': '10px'})])
                    ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                              'marginBottom': '10px'}),
                ], width=6),

                dbc.Col([
                    html.Div([
                        html.Div("TIC PCA (TIC binned in 3s intervals)", style={'textAlign': 'center', 'padding': '5px',
                                                    'backgroundColor': 'rgba(128, 128, 128, 0.1)'}),
                        dcc.Loading(
                            id="loading-pca",
                            type="circle",
                            children=[
                                dcc.Graph(id='pca-plot', style={'height': '600px', 'marginBottom': '10px'})])
                    ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                              'marginBottom': '10px'}),
                ], width=6),
            ]),
        ])


from dash import Input, Output, State


@app.callback(
    Output('mzml-checklist', 'value'),
    [Input('select-mzmls-button', 'n_clicks'),
     Input('unselect-mzmls-button', 'n_clicks')],
    [State('text-field', 'value'),
     State('mzml-checklist', 'value')]
)
def update_checklist(select_clicks, unselect_clicks, text_value, selected_mzmls):
    ctx = dash.callback_context

    if not ctx.triggered_id or selected_mzmls is None:
        raise dash.exceptions.PreventUpdate

    new_selected_mzmls = set(selected_mzmls)

    if text_value is None:
        if 'select-mzmls-button' == ctx.triggered_id:
            return mzml_list
        elif 'unselect-mzmls-button' == ctx.triggered_id:
            return []

    for mzml in mzml_list:
        if text_value.lower() in mzml.lower():
            if 'unselect-mzmls-button' == ctx.triggered_id:
                new_selected_mzmls.discard(mzml)
            elif 'select-mzmls-button' ==  ctx.triggered_id:
                new_selected_mzmls.add(mzml)

    return list(new_selected_mzmls)






@app.callback(
    Output('ms2-scan-counts', 'figure'),
    [Input('mzml-checklist', 'value')]
)
def update_barplot(selected_mzml):

    df = ms2_inv[ms2_inv['mzml_file'].isin(selected_mzml)].copy()
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(by='date_time')
    df['order'] = range(1, len(df) + 1)

    # Create the bar plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['order'],
        y=df['MS2_spectra'],
        name='MS2_spectra',
        hovertemplate=(
            "Order: %{x}<br>"
            "MS2_spectra: %{y}<br>"
            "mzml_file: %{customdata[0]}<br>"
            "Median fragment count: %{customdata[1]}"
        ),
        customdata=df[['mzml_file', 'Median_filtered_Peak_count']].values
    ))

    fig.add_trace(go.Bar(
        x=df['order'],
        y=df['Unique_prec_MZ'],
        name='Unique_prec_MZ',
        hovertemplate=(
            "Order: %{x}<br>"
            "MS2_spectra: %{y}<br>"
            "mzml_file: %{customdata[0]}<br>"
            "Median fragment count: %{customdata[1]}"
        ),
        customdata=df[['mzml_file', 'Median_filtered_Peak_count']].values
    ))

    fig.update_layout(
        height=600,
        barmode='group',
        xaxis=dict(
            showline=True,
            showgrid=False
        ),
        yaxis=dict(
            title="Counts"
        ),
        xaxis_title="Injection order"
    )

    return fig



@app.callback(
    Output('pca-df', 'data'),
    Input('mzml-checklist', 'value'),
    Input('intermediate-value', 'data')
)
def update_pca_dataframes(selected_mzml, make_sure_to_start):
    filtered_ms1_tic = ms1_tic[ms1_tic['mzml_file'].isin(selected_mzml)].copy()

    return_dict = {}

    for col in filtered_ms1_tic.columns:
        if 'TIC_intensity' in col:
            pca_data, loadings_df, top_2_pcs = prepare_pca_data(filtered_ms1_tic, intensity_column=col)

            suffix = col.split('_')[-1]  # Gets the last part after '_' which will be 'complete', '1', '2', etc.

            return_dict[f'pca_data_{suffix}'] = pca_data.to_json()
            return_dict[f'loadings_df_{suffix}'] = loadings_df.to_json()
            return_dict[f'top_2_pcs_{suffix}'] = top_2_pcs

    return json.dumps(return_dict)




@app.callback(
    Output('pca-plot', 'figure'),
    Input('pca-df', 'data')
)

def create_pca_plot(data):

    if data is not None:

        data_dict = json.loads(data)

        # Convert JSON strings back to pandas DataFrames
        pca_data = pd.read_json(data_dict['pca_data_complete'])
        loadings_df = pd.read_json(data_dict['loadings_df_complete'])
        top_2_pcs = data_dict['top_2_pcs_complete']


        unique_dates = pca_data['datetime_order'].unique()
        unique_dates.sort()

        color_map = plt.cm.viridis

        fig = go.Figure()

        # Find renamed columns for top 2 PCs
        renamed_pc1_col = next(col for col in pca_data.columns if top_2_pcs[0] in col)
        renamed_pc2_col = next(col for col in pca_data.columns if top_2_pcs[1] in col)

        for order in sorted(pca_data['datetime_order'].unique()):
            single_order_df = pca_data[pca_data['datetime_order'] == order]

            # Normalize 'order' to [0, 1] range for colormap
            normalized_order = (order - 1) / (len(unique_dates) - 1)
            color_value = [int(x * 255) for x in color_map(normalized_order)[:3]]

            color_str = f'rgb({color_value[0]}, {color_value[1]}, {color_value[2]})'

            fig.add_trace(
                go.Scatter(x=single_order_df[renamed_pc1_col], y=single_order_df[renamed_pc2_col],
                           mode='markers',
                           marker=dict(
                               color=color_str,
                           ),
                           name=f"{order}: {single_order_df['mzml_file'].iloc[0]}",
                           customdata=single_order_df['mzml_file'],
                           hovertemplate="%{customdata}<br>PC1: %{x}<br>PC2: %{y}")
            )

        fig.update_layout(height=600, xaxis_title=renamed_pc1_col, yaxis_title=renamed_pc2_col)

        return fig



@app.callback(
    Output('tic-stats_plot', 'figure'),
    Input('mzml-checklist', 'value'),
    Input('relative-to-median-checkbox', 'value')
)
def TIC_stats_plot(selected_mzml, relative_to_median_checkbox_value):
    filtered_ms1_inv = ms1_inv[ms1_inv['mzml_file'].isin(selected_mzml)].copy()
    filtered_ms1_inv = filtered_ms1_inv.sort_values(by='date_time')

    fig = go.Figure()

    for column in ['TIC_sum', 'TIC_max', 'TIC_median']:

        y_values = filtered_ms1_inv[column]

        if 'show_relative' in relative_to_median_checkbox_value:
            y_values = (y_values / y_values.median() * 100) -100
            y_axis_title = 'Relative to median [%]'
        else:
            y_values = np.log10(filtered_ms1_inv[column])
            y_axis_title = 'Log10(value)'


        fig.add_trace(
            go.Scatter(x=filtered_ms1_inv['date_time'], y=y_values,
                       mode='lines',
                       name=column,
                       customdata=filtered_ms1_inv['mzml_file'],
                       hovertemplate="%{customdata}<br>Date Time: %{x}<br>Log10 Value: %{y}")
        )

    fig.update_layout(height=600,
                      xaxis_title='Date Time',
                      yaxis_title=y_axis_title,
                      yaxis=dict(exponentformat='E'))

    return fig

@app.callback(
    Output('tic-plot', 'figure'),
    Input('mzml-checklist', 'value'),
    Input('pca-df', 'data'),
    Input('pca-plot', 'clickData')
)
def TIC_plot(selected_mzml, pca_dfs, clickData):
    if pca_dfs is None:
        return go.Figure()

    data_dict = json.loads(pca_dfs)
    filtered_ms1_tic = ms1_tic[ms1_tic['mzml_file'].isin(selected_mzml)].copy()
    unique_dates = filtered_ms1_tic['date_time'].unique()
    unique_dates.sort()
    date_order = {date: i + 1 for i, date in enumerate(unique_dates)}
    filtered_ms1_tic['order'] = filtered_ms1_tic['date_time'].map(date_order)

    clicked_trace = None
    clicked_mzml = clickData['points'][0]['customdata'] if clickData else None
    tic_fig = go.Figure()
    color_map = plt.cm.viridis
    max_loading_index = max(
        [int(key.split('_')[-1]) for key in data_dict.keys() if 'loadings_df_' in key and not key.endswith('complete')])
    global_max_tic = filtered_ms1_tic['TIC_intensity_complete'].max()

    norms = []
    for i in range(1, max_loading_index + 1):
        additional_loadings = pd.read_json(data_dict[f'loadings_df_{i}'])
        pc_for_color = data_dict[f'top_2_pcs_{i}'][0]
        pc_loadings = additional_loadings.loc[pc_for_color]
        norm = TwoSlopeNorm(vmin=pc_loadings.min(), vmax=pc_loadings.max(), vcenter=0)

        y0, y1 = ms1_tic_bins.loc[i - 1, 'TIC_bins']

        y0_scaled = (y0 / global_max_tic) * global_max_tic
        y1_scaled = (y1 / global_max_tic) * global_max_tic

        print(y1)

        for rt_bin, loading in pc_loadings.iteritems():
            color_value = cm.RdBu_r(norm(loading))
            color_value_rgb = [int(x * 255) for x in color_value[:3]]
            tic_fig.add_shape(
                type="rect",
                x0=rt_bin,
                x1=rt_bin + 3,
                y0=y0_scaled,
                y1=y1_scaled,
                yref="y",
                fillcolor=f"rgba({color_value_rgb[0]}, {color_value_rgb[1]}, {color_value_rgb[2]}, 0.7)",
                line=dict(width=0),
                layer="below"
            )

    for order in sorted(filtered_ms1_tic['order'].unique()):
        single_order_df = filtered_ms1_tic[filtered_ms1_tic['order'] == order]

        # Normalize 'order' to [0, 1] range for colormap
        normalized_order = (order - 1) / (len(unique_dates) - 1)

        if clicked_mzml and clicked_mzml == single_order_df['mzml_file'].iloc[0]:
            color_value = [255, 0, 0]  # Set to red if clicked
            color_str = f'rgb({color_value[0]}, {color_value[1]}, {color_value[2]})'
            opacity_value = 1
            clicked_trace = go.Scatter(x=single_order_df['rt'], y=single_order_df['TIC_intensity_complete'],
                                       mode='lines',
                                       line=dict(color=color_str),
                                       opacity=opacity_value,
                                       name=f"{order}: {single_order_df['mzml_file'].iloc[0]}",
                                       customdata=single_order_df['mzml_file'],
                                       hovertemplate="%{customdata}<br>Intensity: %{y}")
            continue
        else:
            color_value = [int(x * 255) for x in color_map(normalized_order)[:3]]
            color_str = f'rgb({color_value[0]}, {color_value[1]}, {color_value[2]})'
            if clicked_mzml:
                opacity_value = 0.5
            else:
                opacity_value = 0.7

        single_order_df = single_order_df.copy()
        single_order_df['date_time'] = pd.to_datetime(single_order_df['date_time'])

        tic_fig.add_trace(
            go.Scatter(x=single_order_df['rt'], y=single_order_df['TIC_intensity_complete'],
                       mode='lines',
                       line=dict(
                           color=color_str,
                       ),
                       opacity=opacity_value,
                       name=f"{order}: {single_order_df['mzml_file'].iloc[0]}",
                       customdata=single_order_df['mzml_file'],
                       hovertemplate="%{customdata}<br>Intensity: %{y}")
        )

    if clicked_trace:
        tic_fig.add_trace(clicked_trace)

    tic_fig.update_layout(height=600,
                          xaxis_title='Retention Time',
                          yaxis_title='Intensity',
                          yaxis=dict(exponentformat='E'))



    return tic_fig



# @app.callback(
#     Output('mzml-checklist', 'options'),
#     Input('set-dropdown', 'value')
# )
# def update_mzml_checklist(selected_set):
#     # Sort mzml_list based on time_of_upload
#     df_sorted = ms1_inv.sort_values('date_time')
#     mzml_list_sorted = df_sorted['mzml_file'].unique().tolist()
#
#     sorted_options = []
#     for i, filename in enumerate(mzml_list_sorted):
#         label = f"{i + 1}: {filename}"
#         sorted_options.append({'label': label, 'value': filename})
#
#     return sorted_options

@app.callback(
    Output('subplots', 'figure'),
    Input('mzml-checklist', 'value'),
    Input('set-dropdown', 'value'),
    Input('relative-scale-checkbox', 'value')
)
def update_plots(selected_mzml, selected_set, scale_option):
    filtered_df = df_standards[df_standards['mzml_file'].isin(selected_mzml) & df_standards['collection'].eq(selected_set)]
    filtered_df = filtered_df.sort_values(by='date_time')

    if 'relative' in scale_option:
        median_dict = {}
        for name in filtered_df['name'].unique():
            name_group = filtered_df[filtered_df['name'] == name]
            median_dict[name] = {
                'intensity': name_group['Height'].median(),
                'rt': name_group['RT'].median()
            }

        for name, medians in median_dict.items():
            median_intensity = medians['intensity']
            median_rt = medians['rt']
            name_idx = filtered_df['name'] == name
            filtered_df.loc[name_idx, 'Height'] = ((filtered_df.loc[name_idx, 'Height'] / median_intensity) - 1) * 100
            filtered_df.loc[name_idx, 'RT'] = ((filtered_df.loc[name_idx, 'RT'] / median_rt) - 1) * 100

        yaxis_title_intensity = "Intensity deviation from median [%]"
        yaxis_title_rt = "RT deviation from median [%]"
    else:
        yaxis_title_intensity = "Intensity"
        yaxis_title_rt = "RT"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)



    unique_names = filtered_df['name'].unique()
    color_palette = sns.color_palette("husl", len(unique_names))

    color_map = dict(zip(unique_names, color_palette.as_hex()))

    for index, molecule in enumerate(filtered_df['name'].unique()):
        color_map[molecule] = color_map.get(molecule)

    for molecule in filtered_df['name'].unique():
        molecule_df = filtered_df[filtered_df['name'] == molecule]

        fig.add_trace(
            go.Scatter(x=molecule_df['date_time'], y=molecule_df['Height'],
                       name=molecule, legendgroup=molecule,
                       line=dict(color=color_map[molecule]),
                       hovertemplate='File: %{text}<br>Intensity: %{y}',
                       text=molecule_df['mzml_file'].tolist(),
                       showlegend=True),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=molecule_df['date_time'], y=molecule_df['RT'],
                       name=molecule, legendgroup=molecule,
                       line=dict(color=color_map[molecule]),
                       hovertemplate='File: %{text}<br>RT: %{y}',
                       text=molecule_df['mzml_file'].tolist(),
                       showlegend=False),
            row=2, col=1
        )

    fig.update_layout(
        yaxis_title=yaxis_title_intensity,
        yaxis2_title=yaxis_title_rt,
        height=800,
        legend_title_text='',
        legend=dict(x=0.5, xanchor='center', y=1.1, orientation='h'),
        xaxis2=dict(title='Sample Injection Time'),
        yaxis=dict(exponentformat='E')
    )

    return fig

@app.callback(
    Output('store-df', 'data'),
    [Input('update-button', 'n_clicks')],
    [State('peak-count-slider', 'value'),
     State('purity-slider', 'value'),
     State('intensity-ratio-slider', 'value'),
     State('collision-energy-slider', 'value')]
)
def update_dataframe(n_clicks, peak_count_threshold, purity_threshold, intensity_ratio_threshold, collision_energy_ratio):
    updated_df = assign_MS2_groups(ms2_scans, peak_count_threshold, purity_threshold, intensity_ratio_threshold, collision_energy_ratio)
    return updated_df.to_dict('records')


@app.callback(
    Output('ms2-plot2', 'figure'),
    Input('store-df', 'data'),
    Input('mzml-file-dropdown', 'value'),
    Input('MS2-map-x-dropdown', 'value'),
    Input('MS2-map-y-dropdown', 'value'),
    Input('log10-x-checkbox', 'value'),
    Input('log10-y-checkbox', 'value'),
    State('peak-count-slider', 'value'),
    State('purity-slider', 'value'),
    State('intensity-ratio-slider', 'value'),
    State('collision-energy-slider', 'value')
)

def update_right_plot(df_dict, selected_mzml, plot_x, plot_y, log_x, log_y, peak_count_threshold, purity_threshold, intensity_ratio_threshold, collision_energy_ratio):
    if df_dict is None:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(df_dict)
    return make_MS2_map(df, selected_mzml, plot_x, plot_y, log_x, log_y, peak_count_threshold, purity_threshold, intensity_ratio_threshold, collision_energy_ratio)


def make_MS2_map(df_input, selected_mzml, plot_x, plot_y, log_x, log_y, peak_count_threshold, purity_threshold, intensity_ratio_threshold, collision_energy_ratio):

    df = df_input.copy()

    df = df[df['mzml_file'] == selected_mzml]

    df = assign_MS2_groups(df, peak_count_threshold, purity_threshold, intensity_ratio_threshold, collision_energy_ratio)



    if plot_x in 'Apex/Precursor intensity' or plot_y in 'Apex/Precursor intensity':
        df['Apex/Precursor intensity'] = df.apply(
            lambda row: row['Precursor intensity'] if pd.notna(row['Precursor intensity']) and row[
                'Precursor intensity'] != 0 else row['Feature Apex intensity'],
            axis=1
        )

    if 'log10' in log_x:
        df[f'log({plot_x})'] = np.where(df[plot_x] > 0, np.log10(df[plot_x]), None)
        plot_x = f'log({plot_x})'

    if 'log10' in log_y:
        df[f'log({plot_y})'] = np.where(df[plot_y] > 0, np.log10(df[plot_y]), None)
        plot_y = f'log({plot_y})'

    color_map = {
        'vacant MS2': '#E41A1C',
        'no MS2': '#377EB8',
        f'≥{str(peak_count_threshold)} fragments': '#4DAF4A',
        'redundant MS2': '#984EA3',
        f'<{str(peak_count_threshold)} fragments; precursor purity <{(purity_threshold) * 100}%': '#FF7F00',
        f'<{str(peak_count_threshold)} fragments; precursor ion >{str(collision_energy_ratio * 100)}% of MS2 base peak': '#000000',
        f'<{str(peak_count_threshold)} fragments': '#A65628',
        f'<{str(peak_count_threshold)} fragments; triggered <{str(intensity_ratio_threshold * 100)}% of apex': '#F781BF'
    }

    filtered_color_map = {k: v for k, v in color_map.items() if k in df['f_MS2'].unique()}

    # Initialize an empty figure
    fig = go.Figure()

    # Loop through each category in f_MS2 to plot each as its own scatter series
    for category in df['f_MS2'].unique():
        sub_df = df[df['f_MS2'] == category]
        color = filtered_color_map.get(category, '#808080')

        fig.add_trace(go.Scatter(
            x=sub_df[plot_x],
            y=sub_df[plot_y],
            mode='markers',
            name=category,
            marker=dict(
                color=color,
                line=dict(color=color, width=2),
                size=6,
                opacity=0.7
            )
        ))

    fig.update_layout(
        xaxis_title=plot_x,
        yaxis_title=plot_y
    )

    return fig


# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
