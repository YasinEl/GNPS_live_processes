import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, distinct
import json
import pandas as pd
import argparse
import time

db_path = "C:/Users/elabi/Downloads/aggregated_summary.db"
engine = create_engine(f'sqlite:///{db_path}')

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])


last_mod_time = 0
sorted_options = []
unique_files = []
first_update = True

# Navbar
navbar = dbc.NavbarSimple(
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
                                      style={'overflow': 'auto', 'height': '800px', 'white-space': 'nowrap'}),
                    ], style={'width': '100%'}),
                ])
            ])
            ,
            width=2),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label='Summary', tab_id='summary-tab'),
                dbc.Tab(label='Untargeted', tab_id='untargeted-tab'),
                dbc.Tab(label='Targeted', tab_id='targeted-tab'),
            ], id='tabs', active_tab='untargeted-tab'),
            html.Div(id='tabs-content'),
        ], width=10),
    ]),
    dcc.Interval(
        id='check_file_update',
        interval=2*60*1000,  # check every 2 min
        n_intervals=1
    )
])


def render_tab(tab_value):
    if tab_value == 'untargeted-tab':
        return dbc.Row([
            dbc.Col([
                dbc.Row([
                    html.Div([
                        html.Div("Number and intensity distribution of features with >=2 isotopes.", style={'textAlign': 'center', 'padding': '10px',
                                                         'backgroundColor': 'rgba(0, 128, 128, 0.1)',
                                                         'fontSize': '24px', 'fontWeight': 'bold',
                                                         'marginBottom': '20px'}),
                        html.Div([
                            html.Div(
                                "Numbers of features with and without of MS2 scans",
                                style={'textAlign': 'center', 'padding': '5px',
                                       'backgroundColor': 'rgba(128, 128, 128, 0.1)',
                                       'marginBottom': '20px'}),
                            dcc.Graph(id='feature-counts', style={'height': '400px', 'marginBottom': '10px'}),
                        ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                                  'marginBottom': '10px'}),
                        html.Div([
                            html.Div(
                                "Intensity distribution of features. The red line marks the 3rd percentile of features with MS2 scans. Below that line we only find 3% of all features with MS2 scans.",
                                style={'textAlign': 'center', 'padding': '5px',
                                       'backgroundColor': 'rgba(128, 128, 128, 0.1)',
                                       'marginBottom': '20px'}),
                            dcc.Graph(id='feature-intensities', style={'height': '400px', 'marginBottom': '10px'}),
                        ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                                  'marginBottom': '10px'}),
                    ], style={'border': '2px solid grey', 'border-radius': '12px', 'padding': '15px',
                              'marginBottom': '20px'}),
                ]),
                dbc.Row([
                    html.Div([
                        html.Div("Reasoning for features without MS2", style={'textAlign': 'center', 'padding': '10px',
                                                         'backgroundColor': 'rgba(0, 128, 128, 0.1)',
                                                         'fontSize': '24px', 'fontWeight': 'bold',
                                                         'marginBottom': '20px'}),
                        html.Div([
                            html.Div(
                                "Features lacking MS2 scans: Categorized by intensity relative to 3rd percentile threshold of all features with MS2 scan.",
                                style={'textAlign': 'center', 'padding': '5px',
                                       'backgroundColor': 'rgba(128, 128, 128, 0.1)',
                                       'marginBottom': '20px'}),
                            dcc.Graph(id='features_without_ms2_by_int', style={'height': '400px', 'marginBottom': '10px'}),
                        ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                                  'marginBottom': '10px'}),
                        html.Div([
                            html.Div(
                                "Features lacking MS2 scans above 3rd percentile: Retention time distribution of features with missing MS2 scans.",
                                style={'textAlign': 'center', 'padding': '5px',
                                       'backgroundColor': 'rgba(128, 128, 128, 0.1)',
                                       'marginBottom': '20px'}),
                            dcc.Graph(id='missing-ms2-rt-dist-plot', style={'height': '400px', 'marginBottom': '10px'}),
                        ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                                  'marginBottom': '10px'}),
                        html.Div([
                            html.Div(
                                "Features lacking MS2 scans above 3rd percentile: MS2 scans which have been acquired instead of the missed MS2 scans.",
                                style={'textAlign': 'center', 'padding': '5px',
                                       'backgroundColor': 'rgba(128, 128, 128, 0.1)',
                                       'marginBottom': '20px'}),
                            dcc.Graph(id='reasons_above_int_thr', style={'height': '400px', 'marginBottom': '10px'}),
                        ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                                  'marginBottom': '10px'}),
                        html.Div([
                            html.Div(
                                "Features lacking MS2 scans above 3rd percentile: Origin of vacant and redundant scans triggered while features were missed.",
                                style={'textAlign': 'center', 'padding': '5px',
                                       'backgroundColor': 'rgba(128, 128, 128, 0.1)',
                                       'marginBottom': '20px'}),
                            dcc.Graph(id='vacant-scan-plot', style={'height': '400px', 'marginBottom': '10px'}),
                        ], style={'border': '1px solid grey', 'border-radius': '8px', 'padding': '10px',
                                  'marginBottom': '10px'}),
                        #when looking at rt dimension in 10% bins. where are the missing MS2s

                    ], style={'border': '2px solid grey', 'border-radius': '12px', 'padding': '15px',
                              'marginBottom': '20px'}),
                ])
            ], width=12)
        ], style={'marginTop': '10px'})

    elif tab_value == 'targeted-tab':
        return html.Div([

        ], style={'marginTop': '10px'})

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'active_tab')]
)
def update_tab_content(active_tab):
    return render_tab(active_tab)



@app.callback(
    [Output('mzml-checklist', 'options'),
     Output('mzml-checklist', 'value')],
    [Input('check_file_update', 'n_intervals'),
     Input('select-mzmls-button', 'n_clicks'),
     Input('unselect-mzmls-button', 'n_clicks')],
    [State('text-field', 'value'),
     State('mzml-checklist', 'value'),
     State('mzml-checklist', 'options')]
)
def update_checklist_options(update_check, select_clicks, unselect_clicks, text_value, current_selection, current_options):
    global last_mod_time

    ctx = dash.callback_context

    # Initialize with current values
    updated_options = current_options
    updated_selection = current_selection


    if not ctx.triggered_id and current_options is not None:
        raise dash.exceptions.PreventUpdate

    print('attempt')
    if (ctx.triggered_id and 'check_file_update' in ctx.triggered_id) or current_options is None:

        print('in1')
        current_mod_time = os.path.getmtime(db_path)

        if current_mod_time == last_mod_time:
            raise dash.exceptions.PreventUpdate

        last_mod_time = current_mod_time

        engine = create_engine(f'sqlite:///{db_path}')
        query = "SELECT DISTINCT mzml_file, datetime_order FROM untargetedSummary"
        df_mzmls = pd.read_sql(query, engine)
        engine.dispose()
        #df_mzmls = df_mzmls.sort_values('datetime_order')

        updated_options = [{'label': f"{row['datetime_order']}: {row['mzml_file']}", 'value': row['mzml_file']} for index, row in df_mzmls.iterrows()]

        updated_selection = [option['value'] for option in updated_options]
        if current_options is not None:
            current_option_values = [option['value'] for option in current_options]

            currently_deselected = list(set(current_option_values) - set(current_selection))
            updated_selection = list(set(updated_selection) - set(currently_deselected))

        return updated_options, updated_selection

    elif 'select-mzmls-button' in ctx.triggered_id or 'unselect-mzmls-button.n_clicks' in ctx.triggered_id:
        new_selected_mzmls = set(current_selection)

        if text_value is None:
            if 'select-mzmls-button' == ctx.triggered_id.split('.')[0]:
                updated_selection = [option['value'] for option in current_options]
            elif 'unselect-mzmls-button' == ctx.triggered_id.split('.')[0]:
                updated_selection = []
        else:
            for option in current_options:
                if text_value.lower() in option['label'].lower():
                    if 'unselect-mzmls-button' == ctx.triggered_id.split('.')[0]:
                        new_selected_mzmls.discard(option['value'])
                    elif 'select-mzmls-button' == ctx.triggered_id.split('.')[0]:
                        new_selected_mzmls.add(option['value'])
            updated_selection = list(new_selected_mzmls)
        return updated_options, updated_selection



@app.callback(
    Output('missing-ms2-rt-dist-plot', 'figure'),
    [Input('mzml-checklist', 'value')]
)
def missing_ms2_rt_dist_plot(mzml_checklist):
    if mzml_checklist is None:
        raise dash.exceptions.PreventUpdate

    engine = create_engine(f'sqlite:///{db_path}')
    mzml_placeholders = ', '.join(['?'] * len(mzml_checklist))
    query = f"SELECT mzml_file, datetime_order, Missed_triggers_above_percentile_perRT_10p, last_MS1_scan_rt " \
            f"FROM untargetedSummary " \
            f"WHERE mzml_file IN ({mzml_placeholders})"

    df_feature_count = pd.read_sql(query, engine, params=tuple(mzml_checklist))
    engine.dispose()

    df_feature_count['datetime_order_u'] = range(1, len(df_feature_count) + 1)

    # Convert the comma-separated counts into a list of integers
    df_feature_count['Missed_triggers_above_percentile_perRT_10p'] = df_feature_count['Missed_triggers_above_percentile_perRT_10p'].apply(
        lambda x: list(map(int, x.split(','))))

    # Create a new DataFrame for plotting
    plot_df = pd.DataFrame(columns=['datetime_order_u', 'bin_midpoint', 'mzml_file'])

    # Temporary list to hold new rows
    new_rows = []

    for i, row in df_feature_count.iterrows():
        max_rt = row['last_MS1_scan_rt']
        interval = max_rt * 0.1
        mzml_file = row['mzml_file']

        for j, val in enumerate(row['Missed_triggers_above_percentile_perRT_10p']):
            bin_start = j * interval
            bin_end = (j + 1) * interval
            bin_midpoint = (bin_start + bin_end) / 2

            new_row = {
                'datetime_order_u': row['datetime_order_u'],
                'bin_midpoint': bin_midpoint,
                'mzml_file': mzml_file
            }

            # Duplicate the row based on its value (precomputed count)
            new_rows.extend([new_row] * val)

    # Convert list of new rows to DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    new_rows_df = new_rows_df.dropna(how='all')

    new_rows_df = new_rows_df.dropna(axis=1, how='all')
    plot_df = plot_df.dropna(axis=1, how='all')


    # Concatenate to original DataFrame
    plot_df = pd.concat([plot_df, new_rows_df], ignore_index=True)
    plot_df['bin_midpoint'] = plot_df['bin_midpoint'] /60

    # Create the violin plot
    fig = px.violin(
        plot_df,
        x='datetime_order_u',
        y='bin_midpoint',
        box=False,
        points=False,
        hover_data=['mzml_file'],
    )

    # Update the spanmode for each violin
    for trace in fig.data:
        if trace.type == 'violin':
            trace.spanmode = 'hard'

    # Update layout
    fig.update_layout(
        xaxis_title="Injection order",
        yaxis_title="Retention time [min]",
        legend_title_text='',
        margin=dict(t=10),
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h'
        ),
    )

    return fig


@app.callback(
    Output('vacant-scan-plot', 'figure'),
    [Input('mzml-checklist', 'value')]
)
def obstacle_detail_plot(mzml_checklist):
    if mzml_checklist is None:
        raise dash.exceptions.PreventUpdate

    engine = create_engine(f'sqlite:///{db_path}')
    mzml_placeholders = ', '.join(['?'] * len(mzml_checklist))
    query = f"SELECT mzml_file, datetime_order, Percentage_of_vacant_obstacles_more_than_10, Percentage_of_redundant_scans_after_apex " \
            f"FROM untargetedSummary " \
            f"WHERE mzml_file IN ({mzml_placeholders})"

    df_feature_count = pd.read_sql(query, engine, params=tuple(mzml_checklist))
    engine.dispose()

    df_feature_count['datetime_order_u'] = range(1, len(df_feature_count) + 1)

    # Rename columns for better readability
    df_feature_count.rename(columns={
        'Percentage_of_vacant_obstacles_more_than_10': 'Vacant scans of precursor MZs tiggered >= 10 times [%]',
        'Percentage_of_redundant_scans_after_apex': 'Redundant scans triggered after after feature apex [%]'
    }, inplace=True)

    # Create the figure
    fig = go.Figure()

    # Add traces for each variable
    fig.add_trace(go.Scatter(x=df_feature_count['datetime_order_u'], y=df_feature_count['Vacant scans of precursor MZs tiggered >= 10 times [%]'],
                             mode='lines', name='Vacant scans of precursor MZs tiggered >= 10 times [%]',
                             hovertext=df_feature_count['mzml_file']))

    fig.add_trace(go.Scatter(x=df_feature_count['datetime_order_u'], y=df_feature_count['Redundant scans triggered after after feature apex [%]'],
                             mode='lines', name='Redundant scans triggered after after feature apex [%]',
                             hovertext=df_feature_count['mzml_file']))

    # Update layout
    fig.update_layout(
        xaxis_title="Injection order",
        yaxis_title="Value [%]",
        legend_title_text='',
        margin=dict(t=10),
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h'
        ),
    )

    return fig


@app.callback(
    Output('reasons_above_int_thr', 'figure'),
    [Input('mzml-checklist', 'value')]
)
def get_missingMS2_obstacles_above_thr(mzml_checklist):
    if mzml_checklist is None:
        raise dash.exceptions.PreventUpdate

    engine = create_engine(f'sqlite:///{db_path}')
    mzml_placeholders = ', '.join(['?'] * len(mzml_checklist))
    query = f"SELECT mzml_file, datetime_order, Obstacles_above_percentile_redundant_scans, Obstacels_above_percentile_vacant_scans, Obstacels_above_percentile_successfull_scans, Obstacels_above_percentile_No_scans " \
            f"FROM untargetedSummary " \
            f"WHERE mzml_file IN ({mzml_placeholders})"
    df_feature_count = pd.read_sql(query, engine, params=tuple(mzml_checklist))
    engine.dispose()

    # Sort the DataFrame by datetime_order
    df_feature_count = df_feature_count.sort_values('datetime_order')


    df_feature_count['datetime_order_u'] = range(1, len(df_feature_count) + 1)

    # Rename columns for better readability
    df_feature_count.rename(columns={
        'Obstacles_above_percentile_redundant_scans': 'Redundant MS2s of features',
        'Obstacels_above_percentile_vacant_scans': 'Scans not associated with any feature',
        'Obstacels_above_percentile_successfull_scans': 'MS2 of other feature',
        'Obstacels_above_percentile_No_scans': 'No MS2 was acquired'
    }, inplace=True)

    # Calculate the sum for each row
    df_feature_count['Total'] = df_feature_count[['Redundant MS2s of features', 'Scans not associated with any feature', 'MS2 of other feature', 'No MS2 was acquired']].sum(axis=1)

    # Calculate the relative percentages
    for col in ['Redundant MS2s of features', 'Scans not associated with any feature', 'MS2 of other feature', 'No MS2 was acquired']:
        df_feature_count[col] = (df_feature_count[col] / df_feature_count['Total']) * 100

    # Create the barplot using Plotly
    fig = px.bar(df_feature_count, x='datetime_order_u', y=['Redundant MS2s of features', 'Scans not associated with any feature', 'MS2 of other feature', 'No MS2 was acquired'], hover_data=['mzml_file'])

    fig.update_layout(
        xaxis_title="Injection order",
        yaxis_title="Relative Percentage",
        legend_title_text='',
        yaxis=dict(tickvals=list(range(0, 101, 10)), ticktext=[f"{i}%" for i in range(0, 101, 10)]),
        margin=dict(t=10),
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h'
        ),
    )

    return fig


@app.callback(
    Output('features_without_ms2_by_int', 'figure'),
    [Input('mzml-checklist', 'value')]
)
def features_without_ms2_by_int(mzml_checklist):
    if mzml_checklist is None:
        raise dash.exceptions.PreventUpdate

    engine = create_engine(f'sqlite:///{db_path}')
    mzml_placeholders = ', '.join(['?'] * len(mzml_checklist))
    query = f"SELECT mzml_file, datetime_order, Feature_count, Missed_triggers_above_percentile, Missed_triggers_below_percentile " \
            f"FROM untargetedSummary " \
            f"WHERE mzml_file IN ({mzml_placeholders})"

    df_feature_count = pd.read_sql(query, engine, params=tuple(mzml_checklist))
    engine.dispose()


    df_feature_count['datetime_order_u'] = range(1, len(df_feature_count) + 1)

    # Rename columns for better readability
    df_feature_count.rename(columns={
        'Missed_triggers_above_percentile': 'Missed MS2s > 3rd Percentile',
        'Missed_triggers_below_percentile': 'Missed MS2s < 3rd Percentile'
    }, inplace=True)


    # Create the barplot using Plotly
    fig = px.bar(df_feature_count, x='datetime_order_u',
                 y=['Missed MS2s > 3rd Percentile', 'Missed MS2s < 3rd Percentile'], hover_data=['mzml_file'])

    # Update colors and make Missed_triggers_below_percentile bars negative
    fig.for_each_trace(lambda trace: trace.update(
        marker_color='blue') if trace.name == 'Missed MS2s > 3rd Percentile' else trace.update(marker_color='red',
                                                                                               y=[-val for val in
                                                                                                  trace.y]))

    max_y_value = max(abs(df_feature_count['Missed MS2s > 3rd Percentile']).max(),
                      abs(df_feature_count['Missed MS2s < 3rd Percentile']).max()) * 1.2

    fig.update_layout(
        xaxis_title="Injection order",
        yaxis_title="Features without MS2 [count]",
        legend_title_text='',
        margin=dict(t=10),
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h'
        ),
        yaxis=dict(range=[-max_y_value, max_y_value])
    )

    return fig

@app.callback(
    Output('feature-counts', 'figure'),
    [Input('mzml-checklist', 'value')]
)
def feature_count_bars(mzml_checklist):
    if mzml_checklist is None:
        raise dash.exceptions.PreventUpdate

    engine = create_engine(f'sqlite:///{db_path}')
    mzml_placeholders = ', '.join(['?'] * len(mzml_checklist))
    query = f"SELECT mzml_file, datetime_order, Feature_count, Triggered_features FROM untargetedSummary WHERE mzml_file IN ({mzml_placeholders})"
    df_feature_count = pd.read_sql(query, engine, params=tuple(mzml_checklist))
    engine.dispose()

    df_feature_count['datetime_order_u'] = range(1, len(df_feature_count) + 1)

    # Calculate the non-triggered features
    df_feature_count['Non_triggered_features'] = df_feature_count['Feature_count'] - df_feature_count['Triggered_features']

    # Rename columns for better readability
    df_feature_count.rename(columns={
        'Triggered_features': 'Features with MS2',
        'Non_triggered_features': 'Features without MS2'
    }, inplace=True)

    # Create the barplot using Plotly
    fig = go.Figure(data=[
        go.Bar(name='Features with MS2', x=df_feature_count['datetime_order_u'],
               y=df_feature_count['Features with MS2'], marker_color='blue', hovertext=df_feature_count['mzml_file']),
        go.Bar(name='Features without MS2', x=df_feature_count['datetime_order_u'],
               y=df_feature_count['Features without MS2'], marker_color='red', hovertext=df_feature_count['mzml_file'])
    ])

    fig.update_layout(
        xaxis_title="Injection order",
        yaxis_title="Feature count",
        legend_title_text='',
        margin=dict(t=10),
        barmode='stack',
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h'
        ),
    )

    return fig


@app.callback(
    Output('feature-intensities', 'figure'),
    [Input('mzml-checklist', 'value')]
)
def feature_intensity_boxplots(mzml_checklist):
    if mzml_checklist is None:
        raise dash.exceptions.PreventUpdate

    engine = create_engine(f'sqlite:///{db_path}')
    mzml_placeholders = ', '.join(['?'] * len(mzml_checklist))
    query = f"SELECT mzml_file, datetime_order, Feature_intensities_Q1, Feature_intensities_Q2, Feature_intensities_Q3, Triggered_lowest_int_percentile, Feature_intensities_P3, Feature_intensities_P97 " \
            f"FROM untargetedSummary " \
            f"WHERE mzml_file IN ({mzml_placeholders})"

    df_feature_intensities = pd.read_sql(query, engine, params=tuple(mzml_checklist))
    engine.dispose()

    df_feature_intensities['datetime_order_u'] = range(1, len(df_feature_intensities) + 1)

    fig = go.Figure()

    for dt_order in df_feature_intensities['datetime_order_u'].unique():
        filtered_df = df_feature_intensities[df_feature_intensities['datetime_order_u'] == dt_order]
        mzml = filtered_df['mzml_file'].tolist()[0]
        q1 = np.log10(filtered_df['Feature_intensities_Q1'].mean())
        q2 = np.log10(filtered_df['Feature_intensities_Q2'].mean())
        q3 = np.log10(filtered_df['Feature_intensities_Q3'].mean())
        p3 = np.log10(filtered_df['Feature_intensities_P3'].mean())
        p97 = np.log10(filtered_df['Feature_intensities_P97'].mean())
        triggered_lowest = np.log10(filtered_df['Triggered_lowest_int_percentile'].mean())

        fig.add_trace(go.Box(
            x=[dt_order],
            name=mzml,
            q1=[q1],
            median=[q2],
            q3=[q3],
            lowerfence=[p3],
            upperfence=[p97],
            boxpoints=False,
            line=dict(color='black'),
            fillcolor='royalblue',
            hoverinfo="x+y+text"
        ))

        # Add red horizontal line
        fig.add_shape(
            type='line',
            x0=dt_order - 0.2,
            x1=dt_order + 0.2,
            y0=triggered_lowest,
            y1=triggered_lowest,
            line=dict(
                color='red',
                width=2
            )
        )

    fig.update_layout(
        xaxis_title="Injection order",
        yaxis_title="log10(Feature intensities)",
        yaxis=dict(
            exponentformat='E'
        ),
        showlegend=False,
        margin=dict(t=10)
    )

    return fig



# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
