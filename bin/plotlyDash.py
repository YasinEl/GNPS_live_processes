import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pandas as pd
import matplotlib.pyplot as plt

from plotly.colors import find_intermediate_color
import plotly.express as px
from plotly.express.colors import sample_colorscale

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

# Dummy Data
# dummy_list = [f'File{i}' for i in range(1, 21)]
# unique_dates = pd.date_range('2022-01-01', periods=50, freq='D')
# unique_files = dummy_list[:5]  # Let's take the first 5 for simplicity

# rows = []
# for date in unique_dates:
#     for file in unique_files:
#         for set_name in ['Set1', 'Set2', 'Set3']:
#             for molecule in ['Molecule1', 'Molecule2', 'Molecule3']:
#                 height = np.random.rand()
#                 retention_time = np.random.rand() * 10
#                 rows.append([file, date, set_name, molecule, height, retention_time])
#
# df = pd.DataFrame(rows, columns=['mzml_file', 'date_time', 'set', 'molecule', 'height', 'retention_time'])

df = create_filtered_table('C:/Users/elabi/Downloads/mzml_summary_aggregation.json',  type_="standards")

mzml_list = df['mzml_file'].unique().tolist()


# LC-MS Dummy Data
# lcms_rows = []
# file_date_map = {}
# for index, file in enumerate(unique_files):
#     date_time = unique_dates[index]
#     file_date_map[file] = date_time
#     for molecule in ['Molecule1', 'Molecule2', 'Molecule3']:
#         for peak in np.linspace(1, 10, 10):
#             intensity = np.random.rand() * 1000
#             lcms_rows.append([file, date_time, molecule, peak, intensity])
#
# lcms_df = pd.DataFrame(lcms_rows, columns=['mzml_file', 'date_time', 'molecule', 'retention_time', 'intensity'])

lcms_df = create_filtered_table("C:/Users/elabi/Downloads/mzml_summary_aggregation.json",  type_="standards", include_keys = 'EIC')

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

# def update_lcms_plot(selected_mzml, selected_molecule):
#     filtered_lcms_df = lcms_df[lcms_df['mzml_file'].isin(selected_mzml) & lcms_df['name'].eq(selected_molecule)].copy()
#     filtered_lcms_df['date_time'] = pd.to_datetime(filtered_lcms_df['date_time'])
#
#     date_to_val = filtered_lcms_df['date_time'].map(
#         pd.Series(data=np.arange(len(filtered_lcms_df)), index=filtered_lcms_df['date_time'].values).to_dict())
#
#
#     customdata = [each_date.strftime("%m-%d  %H:%M:%S") for each_date in filtered_lcms_df['date_time']]  # Removed %Y-
#
#     lcms_fig = go.Figure()
#
#     lcms_fig.add_trace(
#         go.Scatter(x=filtered_lcms_df['rt'], y=filtered_lcms_df['intensity'],
#                    mode='lines+markers',
#                    marker=dict(
#                        color=date_to_val,
#                        colorscale='Viridis',
#                        showscale=True,
#                        colorbar=dict(
#                            tickvals=[date_to_val.min(), date_to_val.max()],
#                            ticktext=["First Injection", "Last Injection"],
#                            title_text='Date-Time'
#                        )),
#                    customdata=customdata,
#                    hovertemplate="%{customdata}<br>Intensity: %{y}")
#     )
#
#     lcms_fig.update_layout(height=800, xaxis_title='Retention Time', yaxis_title='Intensity')
#
#     return lcms_fig

def update_lcms_plot(selected_mzml, selected_molecule):
    filtered_lcms_df = lcms_df[lcms_df['mzml_file'].isin(selected_mzml) & lcms_df['name'].eq(selected_molecule)].copy()
    filtered_lcms_df['date_time'] = pd.to_datetime(filtered_lcms_df['date_time'])

    unique_dates = filtered_lcms_df['date_time'].unique()
    unique_dates.sort()
    date_order = {date: i + 1 for i, date in enumerate(unique_dates)}

    filtered_lcms_df['order'] = filtered_lcms_df['date_time'].map(date_order)

    color_map = plt.cm.viridis  # Access colormap directly

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
                    html.Div("Metrics over time", style={'textAlign': 'center', 'padding': '5px', 'backgroundColor': 'rgba(128, 128, 128, 0.1)'}),
                    dcc.Dropdown(id='set-dropdown',
                                 options=[{'label': i, 'value': i} for i in df['collection'].unique()],
                                 value=df['collection'].unique()[0],
                                 style={'width': '200px', 'margin-top': '20px'}),
                    dcc.Graph(id='subplots', style={'height': '800px', 'marginBottom': '10px'})
                ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px', 'marginBottom': '10px'}),
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Div("Extracted ion chromatograms", style={'textAlign': 'center', 'padding': '5px', 'backgroundColor': 'rgba(128, 128, 128, 0.1)'}),
                    dcc.Dropdown(id='molecule-dropdown',
                                 options=[{'label': i, 'value': i} for i in df['name'].unique()],
                                 value=df['name'].unique()[0],
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
        return html.Div([
            # Content for MS1 Tab
        ])


@app.callback(
    Output('mzml-checklist', 'options'),
    Input('set-dropdown', 'value')
)
def update_mzml_checklist(selected_set):
    available_files = df[df['collection'] == selected_set]['mzml_file'].unique().tolist()
    return [{'label': f"{i} {'(Not in Set)' if i not in available_files else ''}", 'value': i} for i in mzml_list]

@app.callback(
    Output('subplots', 'figure'),
    Input('mzml-checklist', 'value'),
    Input('set-dropdown', 'value')
)
def update_plots(selected_mzml, selected_set):
    filtered_df = df[df['mzml_file'].isin(selected_mzml) & df['collection'].eq(selected_set)]
    filtered_df = filtered_df.sort_values(by='date_time')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Intensity", "Retention Time"), vertical_spacing=0.1)

    color_map = {}
    colors = ['red', 'green', 'blue']

    for index, molecule in enumerate(filtered_df['name'].unique()):
        color_map[molecule] = colors[index % len(colors)]

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
        height=800, legend_title_text='Molecule',
        legend=dict(x=0.5, xanchor='center', y=1.1, orientation='h'),
        xaxis2=dict(title='Sample Injection Time'),
        yaxis=dict(title='Intensity'),
        yaxis2=dict(title='Retention Time')
    )

    return fig

# Run app
if __name__ == '__main__':



    app.run_server(debug=True)
