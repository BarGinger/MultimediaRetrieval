"""
3D Shape Viewer Web App - Step 1
Interactive web-based viewer with file selection and 3D visualization

Last updated: 4/9/2025
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import pandas as pd
import os

class OBJParser:
    """Parser for OBJ 3D mesh files"""
    
    @staticmethod
    def parse_obj_file(filepath):
        """
        Parse OBJ file and return vertices and faces
        
        Parameters:
            - filepath: Path to the .obj file

        Returns:
            - vertices: Nx3 numpy array of vertex coordinates
            - faces: Mx3 numpy array of face vertex indices
        """
        vertices = []
        faces = []
        
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                
                if line.startswith('v '):  # Vertex coordinates
                    parts = line.split()
                    if len(parts) >= 4:
                        x, y, z = map(float, parts[1:4])
                        vertices.append([x, y, z])
                        
                elif line.startswith('f '):  # Face indices
                    parts = line.split()
                    if len(parts) >= 4:
                        face_vertices = []
                        for part in parts[1:]:
                            vertex_idx = int(part.split('/')[0]) - 1  # OBJ uses 1-based indexing
                            face_vertices.append(vertex_idx)
                        
                        # Convert to triangles
                        if len(face_vertices) == 3:
                            faces.append(face_vertices)
                        elif len(face_vertices) == 4:  # Quad -> 2 triangles
                            faces.append([face_vertices[0], face_vertices[1], face_vertices[2]])
                            faces.append([face_vertices[0], face_vertices[2], face_vertices[3]])
        
        return np.array(vertices), np.array(faces)

def get_file_tree(data_dir="Data"):
    """
        Get file tree structure for the file browser
        
        Parameters:
            - data_dir: Directory containing the data files
        Returns:
            - df: DataFrame with file information
    """
    files_data = []
    
    # Check if running from src directory, go up to find Data
    current_dir = Path.cwd()
    data_path = current_dir / data_dir
    
    # If not found, try going up one level
    if not data_path.exists():
        data_path = current_dir.parent / data_dir
    
    # If still not found, try going up two levels  
    if not data_path.exists():
        data_path = current_dir.parent.parent / data_dir
    
    print(f"Looking for data in: {data_path}")
    
    if data_path.exists():
        print(f"Found data directory: {data_path}")
        for category_dir in data_path.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                obj_files = list(category_dir.glob("*.obj"))
                print(f"Category '{category}': {len(obj_files)} files")
                for obj_file in obj_files:
                    files_data.append({
                        'category': category,
                        'filename': obj_file.name,
                        'filepath': str(obj_file),
                        'size': obj_file.stat().st_size
                    })
    else:
        print(f"❌ Data directory not found: {data_path}")
        print(f"Current working directory: {current_dir}")
        print("Available directories:")
        for item in current_dir.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}")
    
    df = pd.DataFrame(files_data)
    print(f"Total files found: {len(df)}")
    return df

def create_wireframe_edges(vertices, faces):
    """
    Optimized wireframe edge creation

    Parameters:
        - vertices: Nx3 numpy array of vertex coordinates
        - faces: Mx3 numpy array of face vertex indices
    Returns:
        - wireframe_x, wireframe_y, wireframe_z: Lists of coordinates for wireframe edges
    """
    if len(faces) == 0:
        return [], [], []
    
    # Use sets to avoid duplicate edges
    edges = set()
    
    # Collect unique edges only
    for face in faces:
        face_len = len(face)
        for i in range(face_len):
            v1, v2 = face[i], face[(i + 1) % face_len]
            # Store edge in consistent order to avoid duplicates
            edge = tuple(sorted([v1, v2]))
            edges.add(edge)
    
    # Convert to coordinate arrays
    wireframe_x, wireframe_y, wireframe_z = [], [], []
    
    for v1_idx, v2_idx in edges:
        if v1_idx < len(vertices) and v2_idx < len(vertices):
            wireframe_x.extend([vertices[v1_idx][0], vertices[v2_idx][0], None])
            wireframe_y.extend([vertices[v1_idx][1], vertices[v2_idx][1], None])
            wireframe_z.extend([vertices[v1_idx][2], vertices[v2_idx][2], None])
    
    return wireframe_x, wireframe_y, wireframe_z

def create_3d_plot(vertices, faces, title="3D Shape", show_wireframe=False, mesh_color='lightblue'):
    """
        Create 3D plotly figure with optional wireframe and custom color
    
        Parameters:
            - vertices: Nx3 numpy array of vertex coordinates
            - faces: Mx3 numpy array of face vertex indices
            - title: Title of the plot
            - show_wireframe: Boolean to toggle wireframe display
            - mesh_color: Color of the mesh (default: 'lightblue')
        Returns:
            - fig: Plotly Figure object
    """
    fig = go.Figure()
    
    if len(vertices) == 0:
        fig.add_annotation(text="No data to display", showarrow=False)
        return fig
    
    if len(faces) > 0:
        x, y, z = vertices.T
        i, j, k = faces.T
        
        # Add main mesh
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=mesh_color,
            opacity=0.7,
            name="Mesh",
            lighting=dict(
                ambient=0.18,
                diffuse=1,
                fresnel=0.1,
                specular=1,
                roughness=0.05
            ),
            lightposition=dict(x=100, y=200, z=0)
        ))
        
        # Add wireframe if requested
        if show_wireframe:
            wireframe_x, wireframe_y, wireframe_z = create_wireframe_edges(vertices, faces)
            fig.add_trace(go.Scatter3d(
                x=wireframe_x,
                y=wireframe_y,
                z=wireframe_z,
                mode='lines',
                line=dict(color='black', width=2),
                name="Wireframe",
                hoverinfo='skip'
            ))
    else:
        # Point cloud fallback
        x, y, z = vertices.T
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color='lightblue'),
            name="Point Cloud"
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "3D Shape Viewer"


# Get file data
file_df = get_file_tree()

# Load analysis results for average/outlier selection

# Try both possible paths for analysis_results.csv

# Use the absolute path based on workspace structure
csv_path = os.path.join(Path.cwd(), 'Preprocessing', 'analysis_results.csv')
print(f"Checking for analysis_results.csv at: {csv_path}")

if os.path.exists(csv_path):
    print(f"Found analysis_results.csv at: {csv_path}")
    analysis_csv_path = csv_path
else:
    print("❌ analysis_results.csv not found at expected location!")
    analysis_csv_path = None

analysis_df = pd.read_csv(analysis_csv_path) if analysis_csv_path is not None else None

# Debug: Print first few entries from both DataFrames to compare filename/category
print("\n--- file_df sample ---")
if not file_df.empty:
    print(file_df[['category', 'filename']].head(5))
else:
    print("file_df is empty!")

print("\n--- analysis_df sample ---")
if analysis_df is not None:
    print(analysis_df[['class', 'shape_file']].head(5))
else:
    print("analysis_df is None!")
def get_special_shapes(df):
    """
    Returns indices in file_df for average shape and outliers
    """
    if df is None or file_df.empty:
        return None, []
    # Find average shape (closest to mean vertex/face count)
    avg_v = df['num_vertices'].mean()
    avg_f = df['num_faces'].mean()
    df['dist_to_avg'] = ((df['num_vertices'] - avg_v)**2 + (df['num_faces'] - avg_f)**2)**0.5
    avg_row = df.loc[df['dist_to_avg'].idxmin()]
    # Outliers: >2 std dev from mean (vertex or face count)
    v_std = df['num_vertices'].std()
    f_std = df['num_faces'].std()
    outlier_rows = df[(df['num_vertices'] > avg_v + 2*v_std) | (df['num_vertices'] < avg_v - 2*v_std) |
                     (df['num_faces'] > avg_f + 2*f_std) | (df['num_faces'] < avg_f - 2*f_std)]
    # Map filenames to file_df indices
    def find_index(row):
        shape_file = row['shape_file']
        class_name = row['class']
        matches = file_df[(file_df['filename'] == shape_file) & (file_df['category'] == class_name)]
        if not matches.empty:
            print(f"Match found: {class_name}/{shape_file} -> index {matches.index[0]}")
            return matches.index[0]
        else:
            print(f"No match for: {class_name}/{shape_file}")
            return None
    print("\n--- get_special_shapes mapping ---")
    print(f"Average shape: {avg_row['class']}/{avg_row['shape_file']}")
    avg_idx = find_index(avg_row)
    # Sort outlier_rows by dist_to_avg descending and select top 10
    outlier_rows_sorted = outlier_rows.sort_values('dist_to_avg', ascending=False).head(10)
    outlier_indices = []
    for _, row in outlier_rows_sorted.iterrows():
        idx = find_index(row)
        if idx is not None:
            outlier_indices.append(idx)
    print(f"Average index: {avg_idx}")
    print(f"Outlier indices (top 10 extreme): {outlier_indices}")
    return avg_idx, outlier_indices

avg_shape_idx, outlier_shape_indices = get_special_shapes(analysis_df)

# App layout
app.layout = html.Div([
    # Store for selected file
    dcc.Store(id='selected-file-store'),
    dcc.Store(id='special-shape-store'),
    html.H1("3D Shape Viewer", style={'textAlign': 'center', 'marginBottom': 30}),

    html.Div([
        # Left panel - File browser
        html.Div([
            html.H3("Select 3D Shape", style={'marginBottom': 20}),
            html.Label("Filter by Category:"),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': 'All Categories', 'value': 'all'}] + 
                        [{'label': cat, 'value': cat} for cat in sorted(file_df['category'].unique())] if not file_df.empty else [],
                value='all',
                style={'marginBottom': 20}
            ),
            dcc.Loading(
                id="loading-files",
                children=[
                    html.Div(id='file-list', style={
                        'height': '500px',
                        'overflowY': 'scroll',
                        'border': '1px solid #ddd',
                        'padding': '10px'
                    })
                ],
                type="default"
            )
        ], style={
            'width': '20%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px'
        }),

        # Middle panel - Shape Analysis
        html.Div([
            html.H3("Shape Analysis", style={
                'margin': '0 0 15px 0',
                'color': '#8e44ad',
                'borderBottom': '2px solid #8e44ad',
                'paddingBottom': '10px'
            }),
            html.Div([
                html.Label("Show special shapes:"),
                dcc.Dropdown(
                    id='special-shape-dropdown',
                    options=(
                        ([{'label': 'Average Shape', 'value': avg_shape_idx}] if avg_shape_idx is not None else []) +
                        [{'label': f'Outlier {i+1}', 'value': idx} for i, idx in enumerate(outlier_shape_indices)]
                    ),
                    placeholder="Select to view...",
                    style={'marginBottom': 20}
                ),
                html.Div(id='special-shape-info')
            ])
        ], style={
            'width': '20%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px',
            'backgroundColor': '#f3f0fa',
            'border': '1px solid #d6c6f2',
            'borderRadius': '8px',
            'marginRight': '2%',
            'height': '600px',
            'overflowY': 'auto'
        }),

        # Right panel - 3D viewer
        html.Div([
            html.Div([
                html.Div([
                    html.H3("📄 Shape Information", style={
                        'margin': '0 0 15px 0', 
                        'color': '#2c3e50',
                        'borderBottom': '2px solid #3498db',
                        'paddingBottom': '10px'
                    }),
                    html.Div(id='shape-info', children=[
                        html.P("🔍 Select a 3D shape from the list to view details", 
                               style={'color': '#7f8c8d', 'fontStyle': 'italic'})
                    ])
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'border': '1px solid #dee2e6',
                    'borderRadius': '8px',
                    'padding': '20px',
                    'width': '20%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'marginRight': '2%',
                    'height': '600px',
                    'overflowY': 'auto'
                }),
                html.Div([
                    html.H3("🎮 3D Visualization", style={
                        'margin': '0 0 15px 0',
                        'color': '#2c3e50',
                        'borderBottom': '2px solid #e74c3c',
                        'paddingBottom': '10px'
                    }),
                    html.Div([
                        html.Label("Display Options:", style={'fontWeight': 'bold', 'marginBottom': '8px'}),
                        html.Div([
                            html.Div([
                                html.Label("Wireframe:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'fontSize': '0.9em'}),
                                dcc.Checklist(
                                    id='display-options',
                                    options=[
                                        {'label': ' Show edges', 'value': 'wireframe'}
                                    ],
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
                                        {'label': '🟤 Peru', 'value': 'peru'}
                                    ],
                                    value='lightblue',
                                    style={'fontSize': '0.85em'}
                                )
                            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                        ], style={'marginBottom': '15px'})
                    ], style={'marginBottom': '15px'}),
                    dcc.Loading(
                        id="loading-3d",
                        children=[
                            dcc.Graph(
                                id='3d-plot',
                                figure=create_3d_plot(np.array([]), np.array([]), "Select a shape to view"),
                                # style={'height': '120px !important', 'width': '100%'}
                                className='three-d-plot'
                            )
                        ],
                        type="cube",
                        color="#e74c3c"
                    )
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'border': '1px solid #dee2e6',
                    'borderRadius': '8px',
                    'padding': '20px',
                    'width': '68%',
                    'display': 'inline-block',
                    'verticalAlign': 'top'
                })
            ])
        ], style={
            'width': '58%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px'
        })
    ])
], style={'fontFamily': 'Arial, sans-serif'})
@app.callback(
    [Output('special-shape-info', 'children'), Output('special-shape-store', 'data')],
    [Input('special-shape-dropdown', 'value')],
    prevent_initial_call=True
)
def show_special_shape(selected_idx):
    if selected_idx is None or file_df.empty:
        return dash.no_update, dash.no_update
    try:
        if selected_idx < 0 or selected_idx >= len(file_df):
            raise IndexError(f"Selected index {selected_idx} is out of bounds for file_df (len={len(file_df)})")
        file_info = file_df.iloc[selected_idx]
        filename = file_info['filename']
        category = file_info['category']
        file_size = file_info['size']
        # Parse OBJ file to get actual vertex and face counts
        filepath = file_info['filepath']
        try:
            vertices, faces = OBJParser.parse_obj_file(filepath)
            num_vertices_str = f"{len(vertices):,}"
            num_faces_str = f"{len(faces):,}"
        except Exception as e:
            num_vertices_str = "Error"
            num_faces_str = "Error"
            print(f"Error parsing OBJ file for special shape: {filepath} - {e}")
        info = html.Div([
            html.H4(["⭐ Special Shape: ", filename], style={'color': '#8e44ad', 'marginBottom': '15px'}),
            html.Div([html.Strong("📁 Category: "), category], style={'marginBottom': '8px'}),
            html.Div([html.Strong("💾 File Size: "), f"{file_size:,} bytes"], style={'marginBottom': '8px'}),
            html.Div([html.Strong("🔺 Vertices: "), num_vertices_str], style={'marginBottom': '8px'}),
            html.Div([html.Strong("🔷 Faces: "), num_faces_str], style={'marginBottom': '8px'})
        ])
        return info, selected_idx
    except Exception as e:
        error_info = html.Div([
            html.H4("❌ Error Loading Special Shape", style={'color': '#e74c3c', 'marginBottom': '15px'}),
            html.Div([html.Strong("⚠️ Error: "), str(e)], style={'color': '#e74c3c'})
        ])
        return error_info, dash.no_update

# Callback to update file list based on category filter ONLY
@app.callback(
    Output('file-list', 'children'),
    Input('category-filter', 'value')
)
def update_file_list(selected_category):
    """
    Generate file list - no dependency on selection to avoid reloads
    Parameters:
        - selected_category: Currently selected category filter
    Returns:
        - List of HTML buttons for each file
    """
    if file_df.empty:
        return [html.P("❌ No files found in Data directory", style={'color': 'red', 'textAlign': 'center'})]
    
    filtered_df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
    
    if filtered_df.empty:
        return [html.P(f"❌ No files found for category: {selected_category}", style={'color': 'orange', 'textAlign': 'center'})]
    
    file_buttons = []
    for idx, row in filtered_df.iterrows():
        file_buttons.append(
            html.Button([
                html.Div([
                    html.Strong(f"📁 {row['category']}", className="category-text"),
                    html.Br(),
                    html.Span(f"📄 {row['filename']}", className="filename-text")
                ])
            ],
            id={'type': 'file-btn', 'index': idx},  # Pattern matching ID
            className='file-button',  # Base CSS class only
            n_clicks=0,
            **{'data-file-index': idx}  # Add data attribute for easier JS selection
            )
        )
    
    return file_buttons

# Clientside callback to handle visual selection (runs in browser, no server calls)
app.clientside_callback(
    """
    function(selectedFileIdx) {
        console.log('Selection callback triggered with index:', selectedFileIdx);
        
        if (selectedFileIdx == null || selectedFileIdx === undefined) {
            return window.dash_clientside.no_update;
        }
        
        // Remove selected class from all file buttons
        const allButtons = document.querySelectorAll('[data-file-index]');
        console.log('Found file buttons:', allButtons.length);
        
        allButtons.forEach(button => {
            button.classList.remove('file-button-selected');
        });
        
        // Add selected class to the target button
        const targetButton = document.querySelector(`[data-file-index="${selectedFileIdx}"]`);
        if (targetButton) {
            console.log('Found target button, adding selected class');
            targetButton.classList.add('file-button-selected');
        } else {
            console.log('Target button not found for index:', selectedFileIdx);
        }
        
        return window.dash_clientside.no_update;
    }
    """,
    Output('selected-file-store', 'id'),  # Dummy output
    Input('selected-file-store', 'data')
)

# Create a callback for file button clicks using pattern matching
@app.callback(
    [Output('shape-info', 'children'),
     Output('selected-file-store', 'data')],
    [Input({'type': 'file-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_3d_plot(n_clicks_list):
    """
    Update shape info and selected file index based on button clicks

    Parameters:
        - n_clicks_list: List of n_clicks for all file buttons
    Returns:
        - shape_info: HTML content for shape information
        - selected_file_idx: Index of the selected file
    """

    ctx = dash.callback_context
    
    # Check if there's a triggered event from file buttons
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    # Get the triggered component info
    triggered_info = ctx.triggered[0]
    triggered_id = triggered_info['prop_id']
    triggered_value = triggered_info['value']
    
    # Only proceed if a button was actually clicked (value > 0)
    if triggered_value is None or triggered_value == 0:
        return dash.no_update, dash.no_update
    
    if 'file-btn' not in triggered_id:
        return dash.no_update, dash.no_update
    
    # Extract the file index from the triggered component
    import json
    try:
        component_id = json.loads(triggered_id.split('.')[0])
        file_idx = component_id['index']
        print(f"Button clicked: index {file_idx}, n_clicks: {triggered_value}")
    except Exception as e:
        print(f"Error parsing component ID: {e}")
        return dash.no_update, dash.no_update
    
    if file_idx >= len(file_df):
        return dash.no_update, dash.no_update
    
    # Get the file info
    file_info = file_df.iloc[file_idx]
    filepath = file_info['filepath']
    
    try:
        print(f"Loading file: {filepath}")
        
        # Parse the OBJ file
        vertices, faces = OBJParser.parse_obj_file(filepath)
        
        # Create enhanced shape info
        filename = file_info['filename']
        category = file_info['category']
        file_size = file_info['size']
        
        # Calculate bounding box
        if len(vertices) > 0:
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            dimensions = max_coords - min_coords
        else:
            dimensions = [0, 0, 0]
        
        shape_info = html.Div([
            html.H4([
                "✅ ", filename
            ], style={
                'marginBottom': '15px', 
                'color': '#27ae60',
                'borderBottom': '2px solid #27ae60',
                'paddingBottom': '8px'
            }),
            
            html.Div([
                html.Div([
                    html.Strong("📁 Category: "), category
                ], style={'marginBottom': '8px'}),
                
                html.Div([
                    html.Strong("💾 File Size: "), f"{file_size:,} bytes"
                ], style={'marginBottom': '8px'}),
                
                html.Div([
                    html.Strong("🔺 Vertices: "), f"{len(vertices):,}"
                ], style={'marginBottom': '8px'}),
                
                html.Div([
                    html.Strong("🔷 Faces: "), f"{len(faces):,}"
                ], style={'marginBottom': '8px'}),
                
                html.Div([
                    html.Strong("📐 Dimensions: "), 
                    f"X: {dimensions[0]:.2f}, Y: {dimensions[1]:.2f}, Z: {dimensions[2]:.2f}"
                ], style={'marginBottom': '8px'}),
                
                html.Div([
                    html.Strong("🎯 Quality: "), 
                    "Good" if len(vertices) > 100 and len(faces) > 50 else "Low Resolution"
                ], style={'marginBottom': '8px'})
            ])
        ])
        
        print(f"Successfully loaded: {len(vertices)} vertices, {len(faces)} faces")
        return shape_info, file_idx
        
    except Exception as e:
        print(f"Error loading file {filepath}: {str(e)}")
        
        error_info = html.Div([
            html.H4("❌ Error Loading File", style={'color': '#e74c3c', 'marginBottom': '15px'}),
            
            html.Div([
                html.Strong("📄 File: "), filepath
            ], style={'marginBottom': '8px'}),
            
            html.Div([
                html.Strong("⚠️ Error: "), str(e)
            ], style={'color': '#e74c3c'})
        ])
        
        return error_info, file_idx

@app.callback(
    Output('3d-plot', 'figure'),
    [Input('display-options', 'value'),
     Input('selected-file-store', 'data'),
     Input('special-shape-store', 'data'),
     Input('color-selector', 'value')],
    prevent_initial_call=True
)
def update_3d_visualization(display_options, selected_file_idx, special_shape_idx, mesh_color):
    """
    Update 3D plot based on selected file, special shape, and display options
    """
    # Prioritize special shape selection if present
    idx = special_shape_idx if special_shape_idx is not None else selected_file_idx
    if idx is None:
        return create_3d_plot(np.array([]), np.array([]), "Select a shape to view", mesh_color=mesh_color or 'lightblue')
    try:
        file_info = file_df.iloc[idx]
        filepath = file_info['filepath']
        print(f"Updating 3D visualization for file: {filepath}")
        vertices, faces = OBJParser.parse_obj_file(filepath)
        show_wireframe = 'wireframe' in (display_options or [])
        filename = file_info['filename']
        category = file_info['category']
        fig = create_3d_plot(vertices, faces, f"{category} - {filename}", 
                           show_wireframe=show_wireframe, 
                           mesh_color=mesh_color or 'lightblue')
        print(f"3D plot updated: {len(vertices)} vertices, {len(faces)} faces, wireframe: {show_wireframe}")
        return fig
    except Exception as e:
        print(f"Error updating 3D visualization: {str(e)}")
        return create_3d_plot(np.array([]), np.array([]), "Error loading shape", mesh_color=mesh_color or 'lightblue')

def main():
    """
        Run the web application
        
    """
    print("Starting 3D Shape Viewer...")
    print("Open your browser and go to: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)

if __name__ == "__main__":
    main()
