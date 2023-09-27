import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.colors import find_intermediate_color
import plotly.express as px
from plotly.express.colors import sample_colorscale

import json
import pandas as pd

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



# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])


df_standards = create_filtered_table('C:/Users/elabi/Downloads/mzml_summary_aggregation.json',  type_="standards")

mzml_list = df_standards['mzml_file'].unique().tolist()


lcms_df = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  type_="standards", include_keys = 'EIC')

ms1_inv = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  collection="MS1_inventory")
ms1_tic = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  collection="MS1_inventory", include_keys = 'MS1_inventory')

ms2_inv = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  collection="MS2_inventory")
ms2_scans = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  collection="MS2_inventory", include_keys = 'MS2_inventory')


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
                    dcc.Checklist(id='mzml-checklist',
                                  options=[{'label': i, 'value': i} for i in mzml_list],
                                  value=mzml_list,
                                  style={'overflow': 'auto', 'height': '800px', 'white-space': 'nowrap'}),
                ])
            ]),
            width=2),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Standards', tab_id='standards'),
                dbc.Tab(label='MS2', tab_id='ms2'),
                dbc.Tab(label='MS1', tab_id='ms1'),
            ], id='tabs', active_tab='standards'),
            html.Div(id='tabs-content'),
        ], width=10),
    ])
])




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

    lcms_fig.update_layout(height=800, xaxis_title='Retention Time', yaxis_title='Intensity')

    return lcms_fig


def render_tab(tab_value):
    if tab_value == 'standards':
        return dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Metrics over time", style={'textAlign': 'center', 'padding': '5px',
                                                         'backgroundColor': 'rgba(128, 128, 128, 0.1)'}),
                    html.Div([
                        html.Div([
                            dcc.Dropdown(
                                id='set-dropdown',
                                options=[{'label': i, 'value': i} for i in df_standards['collection'].unique()],
                                value=df_standards['collection'].unique()[0],
                                style={'width': '90%'}
                            ),
                        ], style={'display': 'inline-block', 'width': '30%'}),

                        html.Div([
                            dcc.Checklist(
                                id='relative-scale-checkbox',
                                options=[{'label': 'Relative Scale', 'value': 'relative'}],
                                value=[]
                            ),
                        ], style={'display': 'inline-block', 'width': '20%', 'vertical-align': 'top'}),
                    ], style={'margin-top': '20px'}),

                    dcc.Graph(id='subplots', style={'height': '800px', 'marginBottom': '10px'})
                ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                          'marginBottom': '10px'}),
            ], width=6),

            dbc.Col([
                html.Div([
                    html.Div("Extracted ion chromatograms", style={'textAlign': 'center', 'padding': '5px', 'backgroundColor': 'rgba(128, 128, 128, 0.1)'}),
                    dcc.Dropdown(id='molecule-dropdown',
                                 options=[{'label': i, 'value': i} for i in df_standards['name'].unique()],
                                 value=df_standards['name'].unique()[0],
                                 style={'width': '200px', 'margin-top': '20px'}),
                    dcc.Graph(id='lcms-plot', style={'height': '800px', 'marginBottom': '10px'})
                ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px', 'marginBottom': '10px'}),
            ], width=6)
        ], style={'marginTop': '10px'})
    elif tab_value == 'ms2':
        return html.Div([
            # Content for MS2 Tab
        ])
    elif tab_value == 'ms1':
        return dbc.Row([
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
                    dcc.Graph(id='tic-statistics-plot', style={'height': '800px', 'marginBottom': '10px'})
                ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                          'marginBottom': '10px'}),

            ], width=6),

            dbc.Col([
                html.Div([
                    html.Div("Total ion chromatograms", style={'textAlign': 'center', 'padding': '5px',
                                                               'backgroundColor': 'rgba(128, 128, 128, 0.1)'}),
                    dcc.Graph(id='ms1-upper-plot', style={'height': '800px', 'marginBottom': '10px'})
                ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                          'marginBottom': '10px'}),
            ], width=6),
        ], style={'marginTop': '10px'})


# @app.callback(
#     Output('tic-statistics-plot', 'figure'),
#     Input('mzml-checklist', 'value')
# )
@app.callback(
    Output('tic-statistics-plot', 'figure'),
    Input('mzml-checklist', 'value'),
    Input('relative-to-median-checkbox', 'value')
)
def update_ms1_lower_plot(selected_mzml, relative_to_median_checkbox_value):
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

    fig.update_layout(height=400, xaxis_title='Date Time', yaxis_title=y_axis_title)

    return fig

@app.callback(
    Output('ms1-upper-plot', 'figure'),
    Input('mzml-checklist', 'value')
)
def update_ms1_upper_plot(selected_mzml):

    filtered_ms1_tic = ms1_tic[ms1_tic['mzml_file'].isin(selected_mzml)].copy()

    unique_dates = filtered_ms1_tic['date_time'].unique()
    unique_dates.sort()
    date_order = {date: i + 1 for i, date in enumerate(unique_dates)}

    filtered_ms1_tic = filtered_ms1_tic.copy()
    filtered_ms1_tic['order'] = filtered_ms1_tic['date_time'].map(date_order)

    color_map = plt.cm.viridis

    tic_fig = go.Figure()

    for order in sorted(filtered_ms1_tic['order'].unique()):
        single_order_df = filtered_ms1_tic[filtered_ms1_tic['order'] == order]

        # Normalize 'order' to [0, 1] range for colormap
        normalized_order = (order - 1) / (len(unique_dates) - 1)
        color_value = [int(x * 255) for x in color_map(normalized_order)[:3]]

        color_str = f'rgb({color_value[0]}, {color_value[1]}, {color_value[2]})'

        single_order_df = single_order_df.copy()
        single_order_df['date_time'] = pd.to_datetime(single_order_df['date_time'])

        tic_fig.add_trace(
            go.Scatter(x=single_order_df['rt'], y=single_order_df['intensity'],
                       mode='lines',
                       line=dict(
                           color=color_str,
                       ),
                       name=f"Injection {order}",
                       customdata=single_order_df['mzml_file'],
                       hovertemplate="%{customdata}<br>Intensity: %{y}")
        )

    tic_fig.update_layout(height=800, xaxis_title='Retention Time', yaxis_title='Intensity')


    return tic_fig



@app.callback(
    Output('mzml-checklist', 'options'),
    Input('set-dropdown', 'value')
)
# def update_mzml_checklist(selected_set):
#     available_files = df[df['collection'] == selected_set]['mzml_file'].unique().tolist()
#     return [{'label': f"{i} {'(Not in Set)' if i not in available_files else ''}", 'value': i} for i in mzml_list]
def update_mzml_checklist(selected_set):
    # Sort mzml_list based on time_of_upload
    df_sorted = df_standards.sort_values('date_time')
    mzml_list_sorted = df_sorted['mzml_file'].unique().tolist()
    #available_files = df_standards[df_standards['collection'] == selected_set]['mzml_file'].unique().tolist()

    sorted_options = []
    for i, filename in enumerate(mzml_list_sorted):
        label = f"{i + 1}: {filename}"
        #if filename not in available_files:
        #    label += " (Not in Set)"
        sorted_options.append({'label': label, 'value': filename})

    return sorted_options

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
        xaxis2=dict(title='Sample Injection Time')
    )

    return fig


# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
