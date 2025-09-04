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
app = dash.Dash(__name__)
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
            'width': '30%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px'
        }),
        
        # Right panel - 3D viewer
        html.Div([
            html.H3("3D Visualization", style={'marginBottom': 20}),
            
            # Permanent shape info frame
            html.Div([
                html.Div(id='shape-info', children=[
                    html.Div([
                        html.H4("📄 No File Selected", style={
                            'marginBottom': '10px', 
                            'color': '#6c757d',
                            'textAlign': 'center'
                        }),
                        html.P("Click on a file from the left panel to view its details and 3D model.", style={
                            'textAlign': 'center',
                            'color': '#6c757d',
                            'fontStyle': 'italic'
                        })
                    ])
                ])
            ], style={
                'backgroundColor': '#f8f9fa', 
                'padding': '20px', 
                'borderRadius': '12px',
                'border': '2px solid #e9ecef',
                'marginBottom': '20px',
                'minHeight': '120px'
            }),
            
            # Loading indicator for 3D plot
            dcc.Loading(
                id="loading-3d",
                children=[
                    # 3D plot container
                    html.Div([
                        dcc.Graph(
                            id='3d-plot',
                            figure=create_3d_plot(np.array([]), np.array([]), "Select a shape to view"),
                            style={'height': '600px'}
                        )
                    ], style={
                        'border': '2px solid #e9ecef',
                        'borderRadius': '12px',
                        'overflow': 'hidden'
                    })
                ],
                type="cube",
                color="#007bff"
            )
        ], style={
            'width': '65%',
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
            id=f"file-btn-{idx}",  # Simple ID based on original dataframe index
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

# Create a callback for each possible file button
@app.callback(
    [Output('3d-plot', 'figure'),
     Output('shape-info', 'children')],
    [Input(f'file-btn-{i}', 'n_clicks') for i in range(len(file_df))] if not file_df.empty else [Input('category-filter', 'value')],
    prevent_initial_call=True
)
def update_3d_plot(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    # Find which button was clicked
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not triggered_id.startswith('file-btn-'):
        return dash.no_update, dash.no_update
    
    # Extract the file index
    try:
        file_idx = int(triggered_id.replace('file-btn-', ''))
    except:
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
    # Check if any button was clicked
    if not n_clicks_list or not any(n_clicks_list):
        return dash.no_update, dash.no_update, dash.no_update
    
    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Get the button that triggered the callback
    triggered_id = ctx.triggered[0]['prop_id']
    
    # Extract the index from the triggered button
    import re
    match = re.search(r'"index":(\d+)', triggered_id)
    if not match:
        return dash.no_update, dash.no_update, dash.no_update
    
    file_idx = int(match.group(1))
    
    # Get the current filtered dataframe
    current_df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
    
    if file_idx >= len(current_df):
        return dash.no_update, dash.no_update, dash.no_update
    
    # Get the file info
    file_info = current_df.iloc[file_idx]
    filepath = file_info['filepath']
    
    try:
        # Parse the OBJ file
        vertices, faces = OBJParser.parse_obj_file(filepath)
        
        # Create shape info
        filename = file_info['filename']
        category = file_info['category']
        file_size = file_info['size']
        
        shape_info = html.Div([
            html.H4(f"📄 {filename}", style={'marginBottom': '10px', 'color': '#2c3e50'}),
            html.P(f"📁 Category: {category}"),
            html.P(f"� File Size: {file_size:,} bytes"),
            html.P(f"🔺 Vertices: {len(vertices):,}"),
            html.P(f"🔷 Faces: {len(faces):,}"),
        ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '8px'})
        
        # Create 3D plot
        fig = create_3d_plot(vertices, faces, f"{category}/{filename}")
        
        return fig, shape_info, filepath
        
    except Exception as e:
        error_info = html.Div([
            html.H4(f"❌ Error Loading File", style={'color': 'red'}),
            html.P(f"File: {filepath}"),
            html.P(f"Error: {str(e)}"),
        ], style={'backgroundColor': '#f8d7da', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #f5c6cb'})
        
        return create_3d_plot(np.array([]), np.array([]), "Error loading shape"), error_info, None

def main():
    """Run the web application"""
    print("Starting 3D Shape Viewer...")
    print("Open your browser and go to: http://127.0.0.1:8050")
    app.run_server(debug=True, host='127.0.0.1', port=8050)

if __name__ == "__main__":
    main()