from dash import dcc, html
import numpy as np
from core.plotting import create_3d_plot

def _category_options(file_df):
    if file_df.empty:
        return []
    return [{'label': 'All Categories', 'value': 'all'}] + \
           [{'label': cat, 'value': cat} for cat in sorted(file_df['category'].unique())]

def build_layout(file_df, dataset_options, selected_dataset):
    return html.Div([
        html.H1("3D Shape Viewer", style={'textAlign': 'center', 'marginBottom': 30}),

        html.Div([
            html.Label("Select Dataset:"),
            dcc.Dropdown(
                id='dataset-selector',
                options=[{'label': name, 'value': name} for name in dataset_options],
                value=selected_dataset,
                style={'marginBottom': 20}
            ),
            # Left panel: file browser
            html.Div([
                html.H3("Select 3D Shape", style={'marginBottom': 20}),


                html.Label("Show:"),
                dcc.Dropdown(
                    id='average-filter',
                    options=[
                        {'label': 'All Shapes', 'value': 'all'},
                        {'label': 'Average by Faces', 'value': 'avg_faces'},
                        {'label': 'Average by Vertices', 'value': 'avg_vertices'}
                    ],
                    value='all',
                    style={'marginBottom': 10}
                ),

                html.Label("Filter by Category:"),
                dcc.Dropdown(
                    id='category-filter',
                    options=_category_options(file_df),
                    value='all',
                    style={'marginBottom': 20}
                ),

                html.Label("Sort by:"),
                dcc.Dropdown(
                    id='sort-field',
                    options=[
                        {'label': 'Category', 'value': 'category'},
                        {'label': 'Vertex Count', 'value': 'num_vertices'},
                        {'label': 'Face Count', 'value': 'num_faces'}
                    ],
                    value='category',
                    style={'marginBottom': 10}
                ),

                html.Label("Order:"),
                dcc.Dropdown(
                    id='sort-order',
                    options=[
                        {'label': 'Ascending', 'value': 'asc'},
                        {'label': 'Descending', 'value': 'desc'}
                    ],
                    value='asc',
                    style={'marginBottom': 20}
                ),

                dcc.Loading(
                    id="loading-files",
                    children=[html.Div(
                        id='file-list',
                        style={'height': '600px', 'overflowY': 'scroll',
                               'border': '1px solid #ddd', 'padding': '10px'}
                    )],
                    type="default"
                )
            ], style={'width': '20%', 'display': 'inline-block',
                      'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                      'border': '1px solid #dee2e6', 'borderRadius': '8px'}),

            # Right panel: info + 3D
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("📄 Shape Information", style={
                            'margin': '0 0 15px 0', 'color': '#2c3e50',
                            'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'
                        }),
                        html.Div(id='shape-info', children=[
                            html.P("🔍 Select a 3D shape from the list to view details",
                                   style={'color': '#7f8c8d', 'fontStyle': 'italic'})
                        ])
                    ], style={
                        'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6',
                        'borderRadius': '8px', 'padding': '20px', 'width': '20%',
                        'display': 'inline-block', 'verticalAlign': 'top',
                        'marginRight': '2%', 'height': '700px', 'overflowY': 'auto'
                    }),

                    html.Div([
                        html.H3("🎮 3D Visualization", style={
                            'margin': '0 0 15px 0', 'color': '#2c3e50',
                            'borderBottom': '2px solid #e74c3c', 'paddingBottom': '10px'
                        }),

                        html.Div([
                            html.Label("Display Options:", style={'fontWeight': 'bold', 'marginBottom': '8px'}),
                            html.Div([
                                html.Div([
                                    html.Label("Wireframe:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '0.9em'}),
                                    dcc.Checklist(
                                        id='display-options',
                                        options=[{'label': ' Show edges', 'value': 'wireframe'}],
                                        value=[],
                                        style={'marginTop': '5px'}
                                    )
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                html.Div([
                                    html.Label("Shape Color:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '0.9em'}),
                                    dcc.Dropdown(
                                        id='color-selector',
                                        options=[
                                            {'label': '🔵 Light Blue', 'value': 'lightblue'},
                                            {'label': '🔴 Light Coral', 'value': 'lightcoral'},
                                            {'label': '🟢 Light Green', 'value': 'lightgreen'},
                                            {'label': '🟡 Gold', 'value': 'gold'},
                                            {'label': '🟣 Plum', 'value': 'plum'},
                                            {'label': '🟠 Orange', 'value': 'orange'},
                                            {'label': '🩵 Cyan', 'value': 'cyan'},
                                            {'label': '🩷 Pink', 'value': 'pink'},
                                            {'label': '⚪ Silver', 'value': 'silver'},
                                            {'label': '🟤 Peru', 'value': 'peru'},
                                        ],
                                        value='lightblue',
                                        style={'fontSize': '0.85em'}
                                    )
                                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                            ], style={'marginBottom': '15px'})
                        ], style={'marginBottom': '15px'}),

                        dcc.Loading(
                            id="loading-3d",
                            children=[dcc.Graph(
                                id='3d-plot',
                                figure=create_3d_plot(np.array([]), np.array([]), "Select a shape to view"),
                                style={'height': '100%'}
                            )],
                            type="cube",
                            color="#e74c3c"
                        )
                    ], style={
                        'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6',
                        'borderRadius': '8px', 'padding': '20px', 'width': '68%',
                        'display': 'inline-block', 'verticalAlign': 'top'
                    }),
                ])
            ], style={
                'width': '70%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px'
                })
        ]),

        # Bottom panel: similar shapes plots
        html.Div([
            html.H3("📊 Additional Plots", style={
                'margin': '10px 0 8px 0', 'color': '#2c3e50',
                'borderBottom': '2px solid #8e44ad', 'paddingBottom': '6px'
            }),

            html.Button(
                'Find similar shapes',
                id='find-shapes-button',
                n_clicks=0
            ),

            dcc.Slider(
                id='amount-plots-slider',
                min=1, max=10, step=1, value=5,
                marks={1:'1', 2:'2', 3:'3', 4:'4', 5:'5',6:'6',7:'7',8:'8',9:'9',10:'10'},
                tooltip={'always_visible': False}
            ),
            

            html.Div(id='aux-plots-row', style={
                'display': 'flex',
                'flexWrap': 'nowrap',
                'gap': '12px',
                'overflowX': 'auto',
                'padding': '8px 2px',
                'border': '1px solid #dee2e6',
                'borderRadius': '8px',
                'backgroundColor': '#fafafa',
                'height': '400px'
            }),
        ], style={'margin': '10px 20px 20px 20px'})
    ])
