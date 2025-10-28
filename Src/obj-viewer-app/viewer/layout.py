from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import os
from core.plotting import create_3d_plot
import colorsys


def _category_options(file_df):
    if file_df.empty:
        return []
    return [{'label': 'All Categories', 'value': 'all'}] + \
           [{'label': cat, 'value': cat} for cat in sorted(file_df['category'].unique())]

def build_layout(file_df, dataset_options, selected_dataset):
    # Load the analysis results for UnifiedPreprocessed/Data
    analysis_results_path = "Datasets/UnifiedPreprocessed/Data/analysis_results_unifiedPreprocessed_data.csv"
    analysis_df = pd.read_csv(analysis_results_path)

    # Prepare a full category list and color mapping for the legend
    categories_list = [
        'AircraftBuoyant', 'Apartment', 'AquaticAnimal', 'Bed', 'Bicycle', 'Biplane', 'Bird', 'Bookset', 'Bottle',
        'BuildingNonResidential', 'Bus', 'Car', 'Cellphone', 'Chess', 'City', 'ClassicPiano', 'Computer',
        'ComputerKeyboard', 'Cup', 'DeskLamp', 'DeskPhone', 'Door', 'Drum', 'Fish', 'FloorLamp', 'Glasses',
        'Guitar', 'Gun', 'Hand', 'Hat', 'Helicopter', 'House', 'HumanHead', 'Humanoid', 'Insect', 'Jet', 'Knife',
        'MilitaryVehicle', 'Monitor', 'Monoplane', 'Motorcycle', 'Mug', 'MultiSeat', 'Musical_Instrument',
        'NonWheelChair', 'PianoBoard', 'PlantIndoors', 'PlantWildNonTree', 'Quadruped', 'RectangleTable', 'Rocket',
        'RoundTable', 'Shelf', 'Ship', 'Sign', 'Skyscraper', 'Spoon', 'Starship', 'SubmachineGun', 'Sword', 'Tool',
        'Train', 'Tree', 'Truck', 'TruckNonContainer', 'Vase', 'Violin', 'Wheel', 'WheelChair'
    ]

    # Generate stronger distinct colors using golden-ratio spacing + sat/val cycles
    category_color_map = {}
    golden_angle = 137.508
    sats = [0.92, 0.74, 0.56]
    vals = [0.96, 0.82, 0.68]
    for idx, cat in enumerate(categories_list):
        hue = (idx * golden_angle) % 360.0
        sat = sats[idx % len(sats)]
        val = vals[(idx // len(sats)) % len(vals)]
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, sat, val)
        hex_color = '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))
        category_color_map[cat] = hex_color

    # Build legend item children
    legend_items = []
    for cat in categories_list:
        legend_items.append(
            html.Div([
                html.Span(style={
                    'display': 'inline-block',
                    'width': '12px',
                    'height': '12px',
                    'backgroundColor': category_color_map.get(cat),
                    'borderRadius': '6px',
                    'marginRight': '6px',
                    'border': '1px solid rgba(0,0,0,0.08)'
                }),
                html.Span(cat, style={'fontSize': '0.85em'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '10px', 'marginBottom': '6px'})
        )

    # Determine default dataset for the dropdown. Prefer 'UnifiedPreprocessed/Data' when available.
    dataset_default = selected_dataset
    preferred = 'UnifiedPreprocessed/Data'
    if not dataset_default:
        if preferred in dataset_options:
            dataset_default = preferred
        else:
            preferred_path = os.path.join(os.getcwd(), 'Datasets', 'UnifiedPreprocessed', 'Data')
            if os.path.isdir(preferred_path):
                dataset_default = preferred
            else:
                dataset_default = dataset_options[0] if dataset_options else None

    return html.Div([
    html.H1("3D Shape Viewer", className="main-title"),

    # Toast notification system
    html.Div(id="toast-container", className="toast-container", children=[]),
    dcc.Store(id="toast-store", data={"message": "", "type": "", "icon": "", "timestamp": 0}),
    dcc.Interval(id="toast-interval", interval=50, n_intervals=0, disabled=True),
    
    # Step-specific toast system (for 3D viewer overlay)
    dcc.Store(id="step-toast-store", data={"message": "", "type": "", "icon": "", "timestamp": 0}),
    dcc.Interval(id="step-toast-interval", interval=100, n_intervals=0, disabled=True),
    
    # Lazy loading system stores
    dcc.Store(id="file-data-store", data=[]),  # Complete filtered dataset
    dcc.Store(id='global-descriptors-open', data=False),  # Controls visibility of the global descriptors modal
    dcc.Store(id='aux-selected-file-store', data=None),  # Holds aux/sample selection for the aux modal
    dcc.Store(id='aux-descriptors-open', data=False),    # Controls visibility of the aux descriptors modal
    # Hidden Close trigger so callbacks that reference its n_clicks have a stable Input at registration time
    html.Button('Close hidden', id='global-descriptors-hidden-close-trigger', n_clicks=0, style={'display': 'none'}),
    # Hidden close trigger for aux descriptors modal
    html.Button('Close hidden', id='aux-descriptors-hidden-close-trigger', n_clicks=0, style={'display': 'none'}),
    dcc.Store(id="current-batch-store", data={"batch": 0, "batch_size": 150}),  # Current batch info
    dcc.Store(id="scroll-trigger-store", data=None),  # Scroll detection trigger
    
    # Dummy div for clientside callbacks
    html.Div(id="dummy-div", style={'display': 'none'}),
    # Modal placeholder for Global Descriptors (populated by callbacks)
    html.Div(id='global-descriptors-modal', style={'display': 'none'}),
    # Modal placeholder for auxiliary Shape Info (aux modal)
    html.Div(id='aux-descriptors-modal', style={'display': 'none'}),
    # (Modal close is handled via `global-descriptors-open` store and the
    # in-modal Close button; no persistent hidden button required.)

    # Global loading indicator for shape loading
    html.Div([
        html.Div([
            html.Div([
                html.Span("🔄", className="loading-spinner"),
                html.Span("Loading shape...", className="loading-text")
            ], className="loading-content")
        ], className="global-loading-backdrop")
    ], id="global-loading-indicator", className="global-loading-container", style={'display': 'none'}),

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
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),

                    html.Label("Select Dataset:", style={'fontSize': '12px', 'marginBottom': '2px'}),
                    dcc.Dropdown(
                        id='dataset-selector',
                        options=[{'label': name, 'value': name} for name in dataset_options],
                        value=dataset_default,
                        style={'marginBottom': 4, 'width': '100%'}
                    ),

                    html.Label("Filter by Category:", style={'fontSize': '12px', 'marginBottom': '2px'}),
                    dcc.Dropdown(
                        id='category-filter',
                        options=_category_options(file_df),
                        value='all',
                        className="category-dropdown",
                        style={'width': '100%', 'marginBottom': 4}
                    ),

                    html.Label("Filter by Filename:", style={'fontSize': '12px', 'marginTop': '4px', 'marginBottom': '2px'}),
                    dcc.Input(
                        id='filename-filter',
                        type='text',
                        placeholder='Enter pattern (e.g., m*, *153*)',
                        value='',
                        style={
                            'width': '100%',
                            'padding': '4px 6px',
                            'border': '1px solid #ddd',
                            'borderRadius': '4px',
                            'marginBottom': 4,
                            'boxSizing': 'border-box',
                            'fontSize': '12px',
                            'height': '30px'
                        }
                    ),

                    html.Label("Filter by Vertices:", style={'fontSize': '12px', 'marginTop': 4, 'marginBottom': '2px'}),
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
                                placeholder='Vertices',
                                value='',
                                style={
                                    'width': '100%',
                                    'padding': '4px 6px',
                                    'border': '1px solid #ddd',
                                    'borderRadius': '4px',
                                    'boxSizing': 'border-box',
                                    'height': '30px',
                                    'fontSize': '12px'
                                }
                            )
                        ], className='filter-input-wrapper')
                    ], className='filter-row'),

                    html.Label("Filter by Faces:", style={'fontSize': '12px', 'marginTop': 4, 'marginBottom': '2px'}),
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
                                placeholder='Faces',
                                value='',
                                style={
                                    'width': '100%',
                                    'padding': '4px 6px',
                                    'border': '1px solid #ddd',
                                    'borderRadius': '4px',
                                    'boxSizing': 'border-box',
                                    'height': '30px',
                                    'fontSize': '12px'
                                }
                            )
                        ], className='filter-input-wrapper')
                    ], className='filter-row'),

                    html.Label("Sort by:", style={'fontSize': '12px', 'marginTop': 4, 'marginBottom': '2px'}),
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
                    # File count info - completely separate and in top-left
                    html.Div(
                        id='file-count-info',
                        className="file-count-info",
                        children="📊 Loading...",
                        style={
                            'fontSize': '13px',
                            'color': '#3498db',
                            'fontWeight': 'bold',
                            'marginBottom': '8px',
                            'textAlign': 'left',
                            'width': '100%',
                            'position': 'relative'
                        }
                    ),
                    
                    # File list header with action buttons
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
                        ], style={'display': 'flex', 'gap': '5px', 'justifyContent': 'flex-end'})
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

                # File list container with fixed height and button outside scroll area
                html.Div([
                    dcc.Loading(
                        id="loading-files",
                        children=[html.Div(
                            id='file-list',
                            className="file-list-panel"
                        )],
                        type="default"
                    ),
                    
                    # Lazy loading controls - now sibling to loading-files
                    html.Div([
                        # Visible load more button (back to manual loading for now)
                        html.Button(
                            "📥 Load More Files", 
                            id="load-more-btn",
                            className="load-more-button",
                            n_clicks=0,
                            style={'display': 'block', 'margin': '10px auto'},  # Visible button
                            **{'data-has-more': 'true'}  # Initial state - will be updated by callbacks
                        ),
                        # Scroll sentinel - disabled for now
                        html.Div(
                            id="scroll-sentinel",
                            style={'height': '1px', 'margin': '5px 0', 'display': 'none'}  # Hidden
                        ),
                        # Interval component for scroll detection (temporarily disabled)
                        dcc.Interval(
                            id='scroll-interval',
                            interval=60000,  # Very slow to stop the loop temporarily 
                            n_intervals=0,
                            disabled=True  # Disable completely for now
                        )
                    ], className='load-more-container', style={'textAlign': 'center', 'flexShrink': '0', 'padding': '5px 0'})
                ], className='file-list-wrapper', style={
                    'display': 'flex', 
                    'flexDirection': 'column', 
                    'flex': '1', 
                    'minHeight': '0',
                    'maxHeight': 'calc(100vh - 560px)',  # More room for button by reducing wrapper height
                    'overflow': 'hidden'
                })
            ], className="side-panel"),

            # Center panel: 3D Visualization + Shape Info
            html.Div([
                html.Div([
                    html.H3("🎮 3D Visualization", className="panel-title viz-title"),
                    
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
                        html.Div([
                            html.Span("🔺 ", className="shape-info-icon"), html.Span("Vertices: ", className="shape-info-label"), html.Span("N/A", id='shape-vertices', className="shape-info-prop"),
                        ], style={'display': 'none'}),  # hidden placeholders until a shape is selected
                        html.Div([
                            html.Span("🔷 ", className="shape-info-icon"), html.Span("Faces: ", className="shape-info-label"), html.Span("N/A", id='shape-faces', className="shape-info-prop"),
                        ], style={'display': 'none'}),
                    ], className="shape-info-properties", style={'background': '#fff', 'padding': '12px', 'borderRadius': '8px', 'boxShadow': '0 6px 20px rgba(0,0,0,0.06)', 'minHeight': '120px'}),
                    # Inline global descriptors (populated by callbacks) - placed below Shape Info
                    html.Div(id='inline-global-descriptors', children=[], style={'marginTop': '8px', 'marginBottom': '8px'}),
                    html.Div(id="display-options-container", children=[
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

                            # --- Insert two buttons in the middle of the Center panel ---
                            html.Div([
                                html.Button([
                                    html.Span('🌐', style={'marginRight': '4px', 'fontSize': '14px', 'verticalAlign': 'middle'}),
                                    html.Span('Histograms', style={'verticalAlign': 'middle'})
                                ],
                                    id='show-global-descriptors-btn',
                                    n_clicks=0,
                                    className='center-action-btn pretty-action-btn',
                                    style={
                                        'background': 'linear-gradient(90deg, #2563eb 0%, #38bdf8 100%)',
                                        'width': 'auto',
                                        'color': 'white',
                                        'border': 'none',
                                        'borderRadius': '6px',
                                        'padding': '2px 12px',
                                        'fontWeight': 500,
                                        'fontSize': '12px',
                                        'boxShadow': '0 1px 4px rgba(37,99,235,0.10)',
                                        'cursor': 'pointer',
                                        'transition': 'background 0.2s',
                                        'outline': 'none',
                                        'display': 'flex',
                                        'alignItems': 'center',
                                        'gap': '2px',
                                    }
                                ),
                                html.Button([
                                    html.Span('🧬', style={'marginRight': '4px', 'fontSize': '14px', 'verticalAlign': 'middle'}),
                                    html.Span('Cluster', style={'verticalAlign': 'middle'})
                                ],
                                    id='show-clustering-btn',
                                    n_clicks=0,
                                    className='center-action-btn pretty-action-btn',
                                    style={
                                        'background': 'linear-gradient(90deg, #583191 0%, #C3A1F3 100%)',
                                        'width': 'auto',
                                        'color': 'white',
                                        'border': 'none',
                                        'borderRadius': '6px',
                                        'padding': '2px 12px',
                                        'fontWeight': 500,
                                        'fontSize': '12px',
                                        'boxShadow': '0 1px 4px rgba(190,24,93,0.10)',
                                        'cursor': 'pointer',
                                        'transition': 'background 0.2s',
                                        'outline': 'none',
                                        'display': 'flex',
                                        'alignItems': 'center',
                                        'gap': '2px',
                                    }
                                )
                            ], id='center-action-buttons', style={'display': 'none', 'flexDirection': 'row', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'margin': '0 0 0 0', 'gap': '8px'}),


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

                            html.Div([
                                html.Label("Processing Step:", className="display-step-label"),
                                html.Div(
                                    id='step-info-display',
                                    children="Select a processed shape to enable step navigation",
                                    className="step-info-text"
                                ),
                                html.Div([
                                    html.Div("Orig", className="step-label", id="step-label-0"),
                                    html.Div("Mesh", className="step-label", id="step-label-1"),
                                    html.Div("Trans", className="step-label", id="step-label-2"),
                                    html.Div("Align", className="step-label", id="step-label-3"),
                                    html.Div("Flip", className="step-label", id="step-label-4"),
                                    html.Div("Scale", className="step-label", id="step-label-5"),
                                    html.Div("Final", className="step-label", id="step-label-6"),
                                ], className="step-labels"),
                                dcc.Slider(
                                    id='processing-step-slider',
                                    min=0, max=6, step=1, value=5,
                                    marks={},  # Remove built-in marks
                                    tooltip={'always_visible': True, 'placement': 'bottom'},
                                    disabled=True,  # Initially disabled, enabled when processed shape is selected
                                    className="processing-step-slider"
                                )
                            ], id="display-step-panel", className="display-step-panel", style={'display': 'none'}),
                        ], className="display-toolbar"),
                    ], className="display-options-panel"),
                    html.Div([
                        # Step-specific toast container (positioned over 3D viewer)
                        html.Div(id="step-toast-container", className="step-toast-container", children=[]),
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

                # Static legend placed below the amount-plots-slider so it is always visible
                # Build category list and color mapping (same scheme as callbacks)
                html.Div(id='similar-shapes-legend', children=[
                    html.Details([
                        html.Summary('Legend', style={'cursor': 'pointer', 'fontWeight': '600'}),
                        html.Div(legend_items, style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '8px'})
                    ], style={
                        'display': 'block',
                        'width': '100%',
                        'marginTop': '12px',
                        'border': '1px solid rgba(0,0,0,0.08)',
                        'borderRadius': '6px',
                        'backgroundColor': '#fbfbfb',
                        'padding': '6px'
                    })
                ], style={'width': '100%'}),

                dcc.Loading(
                    id='loading-aux-plots',
                    type='circle',
                    color='#2563eb',
                    children=html.Div(
                        id='aux-plots-row',
                        children=[
                            # Content area that will be populated by the server callback
                            html.Div(id='aux-plots-content', className="aux-plots-row")
                        ]
                    ),
                    style={'display': 'block', 'width': '100%', 'minHeight': '60px'}
                ),
            ], className="right-panel"),
        ], className="main-row")
    ])
