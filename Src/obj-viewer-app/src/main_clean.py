"""
3D Shape Viewer Web App - Step 1
Interactive web-based viewer with file selection and 3D visualization
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path
import pandas as pd

class OBJParser:
    """Parser for OBJ 3D mesh files"""
    
    @staticmethod
    def parse_obj_file(filepath):
        """Parse OBJ file and return vertices and faces"""
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
    """Get file tree structure for the file browser"""
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

def create_3d_plot(vertices, faces, title="3D Shape"):
    """Create 3D plotly figure"""
    fig = go.Figure()
    
    if len(vertices) == 0:
        fig.add_annotation(text="No data to display", showarrow=False)
        return fig
    
    if len(faces) > 0:
        x, y, z = vertices.T
        i, j, k = faces.T
        
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='lightblue',
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

# App layout
app.layout = html.Div([
    # Store for selected file
    dcc.Store(id='selected-file-store'),
    
    html.H1("3D Shape Viewer", style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        # Left panel - File browser
        html.Div([
            html.H3("Select 3D Shape", style={'marginBottom': 20}),
            
            # Category filter
            html.Label("Filter by Category:"),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': 'All Categories', 'value': 'all'}] + 
                        [{'label': cat, 'value': cat} for cat in sorted(file_df['category'].unique())] if not file_df.empty else [],
                value='all',
                style={'marginBottom': 20}
            ),
            
            # Loading indicator for file list
            dcc.Loading(
                id="loading-files",
                children=[
                    # File list
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
        
        # Right panel - 3D viewer
        html.Div([
            # Top row with Shape Information and 3D Visualization side by side
            html.Div([
                # Shape Information (left side of right panel)
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
                
                # 3D Visualization (right side of right panel)
                html.Div([
                    html.H3("🎮 3D Visualization", style={
                        'margin': '0 0 15px 0',
                        'color': '#2c3e50',
                        'borderBottom': '2px solid #e74c3c',
                        'paddingBottom': '10px'
                    }),
                    
                    # Loading indicator for 3D plot
                    dcc.Loading(
                        id="loading-3d",
                        children=[
                            dcc.Graph(
                                id='3d-plot',
                                figure=create_3d_plot(np.array([]), np.array([]), "Select a shape to view"),
                                style={'height': '560px'}
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
            'width': '70%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px'
        })
    ])
], style={'fontFamily': 'Arial, sans-serif'})

# Callback to update file list based on category filter
@app.callback(
    Output('file-list', 'children'),
    Input('category-filter', 'value')
)
def update_file_list(selected_category):
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
                    html.Strong(f"📁 {row['category']}"),
                    html.Br(),
                    html.Span(f"📄 {row['filename']}", style={'fontSize': '0.9em', 'color': '#666'})
                ])
            ],
            id={'type': 'file-btn', 'index': idx},  # Pattern matching ID
            style={
                'width': '100%',
                'marginBottom': '8px',
                'padding': '12px',
                'textAlign': 'left',
                'border': '1px solid #ccc',
                'backgroundColor': '#f9f9f9',
                'cursor': 'pointer',
                'borderRadius': '6px',
                'transition': 'all 0.2s'
            },
            n_clicks=0
            )
        )
    
    return file_buttons

# Create a callback for file button clicks using pattern matching
@app.callback(
    [Output('3d-plot', 'figure'),
     Output('shape-info', 'children')],
    [Input({'type': 'file-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_3d_plot(n_clicks_list):
    ctx = dash.callback_context
    
    # Check if there's a triggered event
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
        
        # Create 3D plot
        fig = create_3d_plot(vertices, faces, f"{category} - {filename}")
        
        print(f"Successfully loaded: {len(vertices)} vertices, {len(faces)} faces")
        return fig, shape_info
        
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
        
        return create_3d_plot(np.array([]), np.array([]), "Error loading shape"), error_info

def main():
    """Run the web application"""
    print("Starting 3D Shape Viewer...")
    print("Open your browser and go to: http://127.0.0.1:8050")
    app.run_server(debug=True, host='127.0.0.1', port=8050)

if __name__ == "__main__":
    main()
