from dash import dcc, html
import numpy as np
from core.plotting import create_3d_plot

def _category_options(file_df):
    if file_df.empty:
        return []
    return [{'label': 'All Categories', 'value': 'all'}] + \
           [{'label': cat, 'value': cat} for cat in sorted(file_df['category'].unique())]

def build_layout(file_df):
    return html.Div([
    html.H1("3D Shape Viewer", className="main-title"),

    html.Div([
            # Left panel: file browser
            html.Div([
                html.H3("Select 3D Shape", className="panel-title"),

                html.Label("Filter by Category:"),
                dcc.Dropdown(
                    id='category-filter',
                    options=_category_options(file_df),
                    value='all',
                    className="category-dropdown"
                ),

                dcc.Loading(
                    id="loading-files",
                    children=[html.Div(
                        id='file-list',
                        className="file-list-panel"
                    )],
                    type="default"
                )
            ], className="side-panel"),

            # Center panel: 3D Visualization + Shape Info
            html.Div([
                html.Div([
                    html.H3("🎮 3D Visualization", className="panel-title viz-title"),

                    html.Div([
                        html.H4("📄 Shape Info", className="shape-info-title"),
                        html.Div(id='shape-info', children=[
                            html.P("🔍 Select a 3D shape from the list to view details", className="shape-info-hint"),
                        ], className="shape-info-properties")
                    ], className="shape-info-card"),

                    html.Div([
                        html.Label("Display Options:", className="display-options-label"),
                        html.Div([
                            html.Div([
                                html.Label("Wireframe:", className="display-wireframe-label"),
                                dcc.Checklist(
                                    id='display-options',
                                    options=[{'label': ' Show edges', 'value': 'wireframe'}],
                                    value=[],
                                    className="display-wireframe-checklist"
                                )
                            ], className="display-wireframe-panel"),

                            html.Div([
                                html.Label("Shape Color:", className="display-color-label"),
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
                                    className="display-color-dropdown"
                                )
                            ], className="display-color-panel"),
                        ], className="display-options-row")
                    ], className="display-options-panel"),

                    dcc.Loading(
                        id="loading-3d",
                        children=[dcc.Graph(
                            id='3d-plot',
                            figure=create_3d_plot(np.array([]), np.array([]), "Select a shape to view"),
                            className='main-three-d-plot'
                        )],
                        type="cube",
                        color="#e74c3c"
                    )
                ])
            ], className="center-panel"),

            # Right panel: Additional Plots
            html.Div([
                html.H3("📊 Additional Plots", className="panel-title aux-title"),

                html.Button(
                    'Find similar shapes',
                    id='find-shapes-button',
                    n_clicks=0,
                    className="aux-btn"
                ),

                dcc.Slider(
                    id='amount-plots-slider',
                    min=1, max=10, step=1, value=5,
                    marks={1:'1', 2:'2', 3:'3', 4:'4', 5:'5',6:'6',7:'7',8:'8',9:'9',10:'10'},
                    tooltip={'always_visible': False},
                    className="aux-slider"
                ),

                html.Div(id='aux-plots-row', className="aux-plots-row"),
            ], className="right-panel"),
        ], className="main-row")
    ])
