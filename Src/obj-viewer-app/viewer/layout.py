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
    html.H1("3D Shape Viewer", className="main-title"),

    # Toast notification system
    html.Div(id="toast-container", className="toast-container", children=[]),
    dcc.Store(id="toast-store", data={"message": "", "type": "", "icon": "", "timestamp": 0}),
    dcc.Interval(id="toast-interval", interval=100, n_intervals=0, disabled=True),

    html.Div([            
            # Left panel: file browser
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Select 3D Shape", className="panel-title", style={'margin': 0, 'flex': 1}),
                        html.Button(
                            "🧹",
                            id='clear-filters-btn',
                            title="Clear all filters",
                            style={
                                'backgroundColor': 'rgb(144 198 255)',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '50%',
                                'width': '30px',
                                'height': '30px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'display': 'flex',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'marginLeft': '10px'
                            },
                            n_clicks=0
                        )
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),

                    html.Label("Select Dataset:"),
                    dcc.Dropdown(
                        id='dataset-selector',
                        options=[{'label': name, 'value': name} for name in dataset_options],
                        value=selected_dataset,
                        style={'marginBottom': 20, 'width': '100%'}
                    ),

                    html.Label("Filter by Category:"),
                    dcc.Dropdown(
                        id='category-filter',
                        options=_category_options(file_df),
                        value='all',
                        className="category-dropdown",
                        style={'width': '100%'}
                    ),

                    html.Label("Filter by Filename:", style={'marginTop': 15}),
                    dcc.Input(
                        id='filename-filter',
                        type='text',
                        placeholder='Enter filename pattern (e.g., m*, *153*, bike.obj)',
                        value='',
                        style={
                            'width': '100%',
                            'padding': '8px',
                            'border': '1px solid #ddd',
                            'borderRadius': '4px',
                            'marginBottom': 10,
                            'boxSizing': 'border-box'
                        }
                    ),

                    html.Label("Filter by Vertices:", style={'marginTop': 15, 'marginBottom': 5}),
                    html.Div([
                        html.Div([
                            dcc.Dropdown(
                                id='vertices-operator',
                                options=[
                                    {'label': 'Equal', 'value': 'eq'},
                                    {'label': 'Greater than', 'value': 'gt'},
                                    {'label': 'Less than', 'value': 'lt'}
                                ],
                                value='gt',
                                style={'width': '100%'}
                            )
                        ], className='filter-dropdown-wrapper'),
                        html.Div([
                            dcc.Input(
                                id='vertices-value',
                                type='number',
                                placeholder='Number of vertices',
                                value='',
                                style={
                                    'width': '100%',
                                    'padding': '8px',
                                    'border': '1px solid #ddd',
                                    'borderRadius': '4px',
                                    'boxSizing': 'border-box',
                                    'height': '38px'
                                }
                            )
                        ], className='filter-input-wrapper')
                    ], className='filter-row'),

                    html.Label("Filter by Faces:", style={'marginTop': 15, 'marginBottom': 5}),
                    html.Div([
                        html.Div([
                            dcc.Dropdown(
                                id='faces-operator',
                                options=[
                                    {'label': 'Equal', 'value': 'eq'},
                                    {'label': 'Greater than', 'value': 'gt'},
                                    {'label': 'Less than', 'value': 'lt'}
                                ],
                                value='gt',
                                style={'width': '100%'}
                            )
                        ], className='filter-dropdown-wrapper'),
                        html.Div([
                            dcc.Input(
                                id='faces-value',
                                type='number',
                                placeholder='Number of faces',
                                value='',
                                style={
                                    'width': '100%',
                                    'padding': '8px',
                                    'border': '1px solid #ddd',
                                    'borderRadius': '4px',
                                    'boxSizing': 'border-box',
                                    'height': '38px'
                                }
                            )
                        ], className='filter-input-wrapper')
                    ], className='filter-row'),

                    html.Label("Sort by:"),
                    dcc.Dropdown(
                        id='sort-field',
                        options=[
                            {'label': 'Category', 'value': 'category'},
                            {'label': 'Vertex Count', 'value': 'num_vertices'},
                            {'label': 'Face Count', 'value': 'num_faces'}
                        ],
                        value='category',
                        style={'marginBottom': 10, 'width': '100%'}
                    ),
                ], className="side-panel-controls"),

                html.Div([
                    html.Div([
                        html.Button(
                            "📊",
                            id='avg-vertices-btn',
                            title="Scroll to Average Vertices",
                            className="action-btn",
                            n_clicks=0
                        ),
                        html.Button(
                            "🔷",
                            id='avg-faces-btn',
                            title="Scroll to Average Faces",
                            className="action-btn",
                            n_clicks=0
                        ),
                        html.Button(
                            "↑",
                            id='sort-order',
                            title="Sort Order: Ascending (click to change to Descending)",
                            className="sort-order-btn",
                            n_clicks=0,
                            **{'data-order': 'asc'}
                        )
                    ], className="file-list-buttons"),
                    
                    # Loading indicator for navigation
                    html.Div([
                        html.Div([
                            html.Span("🔍", className="loading-icon"),
                            html.Span("Finding average shape...", className="loading-text")
                        ], className="loading-content")
                    ], id="navigation-loading", className="navigation-loading", style={'display': 'none'}),
                    
                    # Toast message (similar to loading indicator)
                    html.Div([
                        html.Div([
                            html.Span("📊", id="toast-icon", className="toast-icon"),
                            html.Span("Sort order changed", id="toast-message", className="toast-message")
                        ], className="loading-content")
                    ], id="toast-message-bar", className="toast-message-bar", style={'display': 'none'})
                ], className="file-list-header"),

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
                        html.Div([
                            html.H4("📄 Shape Info", className="shape-info-title"),
                            html.Div([
                                html.Label("Normalized:", className="normalization-label", style={'fontSize': '0.85em', 'fontWeight': 'bold', 'marginRight': '8px'}),
                                dcc.Checklist(
                                    id='normalization-toggle',
                                    options=[
                                        {'label': ' Yes', 'value': 'normalized'}
                                    ],
                                    value=[],
                                    className="normalization-checklist",
                                    style={'display': 'inline-block'}
                                )
                            ], style={'marginBottom': '8px', 'paddingBottom': '6px', 'borderBottom': '1px solid #ddd'}),
                        ], style={'marginBottom': '8px'}),
                        html.Div(id='shape-info', children=[
                            html.P("🔍 Select a 3D shape from the list to view details", className="shape-info-hint"),
                        ], className="shape-info-properties")
                    ], className="shape-info-card"),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Wireframe:", className="display-wireframe-label"),
                                dcc.Checklist(
                                    id='display-options',                                    
                                    options=[
                                        {'label': ' Show edges', 'value': 'wireframe'},
                                        {'label': ' Smooth shading', 'value': 'smooth_shading'}
                                    ],
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
                        ], className="display-toolbar"),
                    ], className="display-options-panel"),

                    html.Div([
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
                    ], className="center-plot-container")
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
