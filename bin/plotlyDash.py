import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dummy Data
dummy_list = [f'File{i}' for i in range(1, 21)]
unique_dates = pd.date_range('2022-01-01', periods=50, freq='D')
unique_files = dummy_list[:5]  # Let's take the first 5 for simplicity

rows = []
for date in unique_dates:
    for file in unique_files:
        for set_name in ['Set1', 'Set2', 'Set3']:
            for molecule in ['Molecule1', 'Molecule2', 'Molecule3']:
                height = np.random.rand()
                retention_time = np.random.rand() * 10
                rows.append([file, date, set_name, molecule, height, retention_time])

df = pd.DataFrame(rows, columns=['mzml_file', 'date_time', 'set', 'molecule', 'height', 'retention_time'])

# LC-MS Dummy Data
lcms_rows = []
file_date_map = {}
for index, file in enumerate(unique_files):
    date_time = unique_dates[index]
    file_date_map[file] = date_time
    for molecule in ['Molecule1', 'Molecule2', 'Molecule3']:
        for peak in np.linspace(1, 10, 10):
            intensity = np.random.rand() * 1000
            lcms_rows.append([file, date_time, molecule, peak, intensity])

lcms_df = pd.DataFrame(lcms_rows, columns=['mzml_file', 'date_time', 'molecule', 'retention_time', 'intensity'])


# Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Page 1", href="/")),
    ],
    brand="GNPS-live",
    color="primary",
    dark=True,
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    dbc.Row([
        dbc.Col(
            html.Div([
                html.Div(children='Uploaded files',
                         style={'textAlign': 'center', 'fontSize': 24, 'marginBottom': '10px'}),
                dcc.Checklist(id='mzml-checklist',
                              options=[{'label': i, 'value': i} for i in dummy_list],
                              value=dummy_list,
                              style={'overflow': 'auto', 'height': '800px', 'white-space': 'nowrap'}),
            ]),
            width=2),
        dbc.Col(
            html.Div([
                dcc.Tabs(id='tabs', value='standards', children=[
                    dcc.Tab(label='Standards', value='standards'),
                    dcc.Tab(label='MS2', value='ms2'),
                    dcc.Tab(label='MS1', value='ms1'),
                ]),
                html.Div(id='tabs-content')
            ]),
            width=10)
    ]),
])



@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def update_tab_content(tab_value):
    return render_tab(tab_value)

@app.callback(
    Output('lcms-plot', 'figure'),
    Input('mzml-checklist', 'value'),
    Input('molecule-dropdown', 'value')
)
def update_lcms_plot(selected_mzml, selected_molecule):
    filtered_lcms_df = lcms_df[lcms_df['mzml_file'].isin(selected_mzml) & lcms_df['molecule'].eq(selected_molecule)]

    lcms_fig = go.Figure()

    for date_time in filtered_lcms_df['date_time'].unique():
        date_df = filtered_lcms_df[filtered_lcms_df['date_time'] == date_time]

        lcms_fig.add_trace(
            go.Scatter(x=date_df['retention_time'], y=date_df['intensity'],
                       name=str(date_time),
                       hovertemplate='Date: %{text}<br>Intensity: %{y}',
                       text=date_df['date_time'].tolist(),
                       showlegend=True)
        )

    lcms_fig.update_layout(height=800, legend_title_text='Date-Time', xaxis_title='Retention Time',
                           yaxis_title='Intensity',
                           legend=dict(x=0.5, xanchor='center', y=1.1, orientation='h'))

    return lcms_fig


def render_tab(tab_value):
    if tab_value == 'standards':
        return dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Metrics over time", style={'textAlign': 'center', 'padding': '5px', 'backgroundColor': 'rgba(128, 128, 128, 0.1)'}),
                    dcc.Dropdown(id='set-dropdown',
                                 options=[{'label': i, 'value': i} for i in df['set'].unique()],
                                 value=df['set'].unique()[0],
                                 style={'width': '200px', 'margin-top': '20px'}),
                    dcc.Graph(id='subplots', style={'height': '800px', 'marginBottom': '10px'})
                ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px', 'marginBottom': '10px'}),
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Div("Extracted ion chromatograms", style={'textAlign': 'center', 'padding': '5px', 'backgroundColor': 'rgba(128, 128, 128, 0.1)'}),
                    dcc.Dropdown(id='molecule-dropdown',
                                 options=[{'label': i, 'value': i} for i in df['molecule'].unique()],
                                 value=df['molecule'].unique()[0],
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
    available_files = df[df['set'] == selected_set]['mzml_file'].unique().tolist()
    return [{'label': f"{i} {'(Not in Set)' if i not in available_files else ''}", 'value': i} for i in dummy_list]

@app.callback(
    Output('subplots', 'figure'),
    Input('mzml-checklist', 'value'),
    Input('set-dropdown', 'value')
)
def update_plots(selected_mzml, selected_set):
    filtered_df = df[df['mzml_file'].isin(selected_mzml) & df['set'].eq(selected_set)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Intensity", "Retention Time"), vertical_spacing=0.1)

    color_map = {}
    colors = ['red', 'green', 'blue']

    for index, molecule in enumerate(filtered_df['molecule'].unique()):
        color_map[molecule] = colors[index % len(colors)]

    for molecule in filtered_df['molecule'].unique():
        molecule_df = filtered_df[filtered_df['molecule'] == molecule]

        fig.add_trace(
            go.Scatter(x=molecule_df['date_time'], y=molecule_df['height'],
                       name=molecule, legendgroup=molecule,
                       line=dict(color=color_map[molecule]),
                       hovertemplate='Molecule: %{text}<br>Intensity: %{y}',
                       text=molecule_df['mzml_file'].tolist(),
                       showlegend=True),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=molecule_df['date_time'], y=molecule_df['retention_time'],
                       name=molecule, legendgroup=molecule,
                       line=dict(color=color_map[molecule]),
                       hovertemplate='Molecule: %{text}<br>RT: %{y}',
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
