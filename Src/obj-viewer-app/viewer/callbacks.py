
from dash import dcc, html, Input, Output, no_update, State, callback_context
import dash
import numpy as np
import os
import pandas as pd
import json
from core.obj_parser import OBJParser
from core.plotting import create_3d_plot
import plotly.graph_objects as go
from core.file_index import get_file_tree
from core.analysis_cache import merge_analysis_data
from core.shapeMesh import ShapeMesh




def register_callbacks(app: dash.Dash, file_df, dataset_options, default_dataset):

    # 1) File list render
    @app.callback(
        Output('file-list', 'children'),
        [Input('average-filter', 'value'),
         Input('category-filter', 'value'),
         Input('sort-field', 'value'),
         Input('sort-order', 'value'),
         Input('selected-dataset-store', 'data')]
    )
    def update_file_list(avg_filter, selected_category, sort_field, sort_order, selected_dataset):        
        """
        Render the list of files based on current filters and sorting.

        Parameters:
        - avg_filter: str, average filter option ('none', 'avg_faces', 'avg_vertices')
        - selected_category: str, selected category filter ('all' or specific category)
        - sort_field: str, field to sort by ('category', 'num_vertices', 'num_faces')
        - sort_order: str, sort order ('asc' or 'desc')
        - selected_dataset: str, currently selected dataset from dropdown

        Returns:
        - List of HTML button elements representing the files
        """
        if selected_dataset is None or selected_dataset == "":
            selected_dataset = 'Data'

        file_df = get_file_tree(selected_dataset)

        # Merge analysis CSV columns using cache
        file_df = merge_analysis_data(file_df, selected_dataset)
        if file_df.empty:
            return [html.P("❌ No files found in Data directory",
                           style={'color': 'red', 'textAlign': 'center'})]
        df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]

        ascending = True if sort_order == 'asc' else False
        df = df.copy()
        if sort_field == 'category':
            df = df.sort_values(by=['category', 'filename'], ascending=ascending)
        elif sort_field in ['num_vertices', 'num_faces']:
            # Check if required columns exist for sorting
            if sort_field not in df.columns:
                return [html.Div([
                    html.P("⚠️ Sorting Not Available", style={
                        'color': '#f39c12', 'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '10px'
                    }),
                    html.P(f"Cannot sort by {sort_field} - analysis data not available for this dataset.", style={
                        'color': '#7f8c8d', 'textAlign': 'center', 'marginBottom': '5px'
                    }),
                    html.P("Try sorting by Category instead.", style={
                        'color': '#7f8c8d', 'textAlign': 'center', 'fontSize': '0.9em'
                    })
                ])]
            df[sort_field] = df[sort_field].fillna(0)
            df = df.sort_values(by=sort_field, ascending=ascending)

        # Now apply average filtering after sorting
        if avg_filter == 'avg_faces' and 'num_faces' in df.columns and not df.empty:
            valid = df['num_faces'].dropna()
            if not valid.empty:
                avg_f = valid.mean()
                idx = (df['num_faces'] - avg_f).abs().idxmin()
                if idx in df.index:
                    df = df.loc[[idx]].reset_index(drop=True)
                else:
                    return [html.P("❌ No valid shapes for average by faces", style={'color': 'orange', 'textAlign': 'center'})]
            else:
                return [html.P("❌ No valid shapes for average by faces", style={'color': 'orange', 'textAlign': 'center'})]
        elif avg_filter == 'avg_vertices' and 'num_vertices' in df.columns and not df.empty:
            valid = df['num_vertices'].dropna()
            if not valid.empty:
                avg_v = valid.mean()
                idx = (df['num_vertices'] - avg_v).abs().idxmin()
                if idx in df.index:
                    df = df.loc[[idx]].reset_index(drop=True)
                else:
                    return [html.P("❌ No valid shapes for average by vertices", style={'color': 'orange', 'textAlign': 'center'})]
            else:
                return [html.P("❌ No valid shapes for average by vertices", style={'color': 'orange', 'textAlign': 'center'})]

        # Final reset of index to ensure sequential 0,1,2,... indices for button creation
        df = df.reset_index(drop=True)

        buttons = []
        for btn_idx, (df_idx, row) in enumerate(df.iterrows()):
            # Get vertex and face counts if available
            vertices_count = f"{int(row.get('num_vertices', 0)):,}" if pd.notna(row.get('num_vertices')) else "N/A"
            faces_count = f"{int(row.get('num_faces', 0)):,}" if pd.notna(row.get('num_faces')) else "N/A"
            
            btn = html.Button(
                html.Div([
                    html.Div([
                        html.Strong(f"📁 {row['category']}", className="category-text", style={'fontSize': '0.9em'}),
                        html.Span(f" | 📄 {row['filename']}", className="filename-text", style={'fontSize': '0.85em', 'color': '#555'})
                    ], style={'marginBottom': '2px'}),
                    html.Div([
                        html.Span(f"🔺 Vertices: {vertices_count}", className="stats-text", 
                                style={'marginRight': '8px', 'fontSize': '0.75em', 'color': '#888'}),
                        html.Span(f"🔷 Faces: {faces_count}", className="stats-text", 
                                style={'fontSize': '0.75em', 'color': '#888'})
                    ])
                ]),
                id={'type': 'file-btn', 'index': int(btn_idx)},
                className='file-button',
                n_clicks=0,
                **{'data-file-index': int(btn_idx)}
            )
            buttons.append(btn)
        return buttons

    # 2) Selected file highlight (client-side)
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
        Output('selected-file-store', 'id'),
        Input('selected-file-store', 'data')
    )

    # 3) Click handler -> loads file, updates info + selected index
    @app.callback(
        [Output('shape-info', 'children'),
         Output('selected-file-store', 'data')],
        [Input({'type': 'file-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
         Input('average-filter', 'value'),
         Input('category-filter', 'value'),
         Input('sort-field', 'value'),
         Input('sort-order', 'value'),
         Input('selected-dataset-store', 'data')],
        prevent_initial_call=True
    )
    def select_or_reset_file(n_clicks_list, avg_filter, selected_category, sort_field, sort_order, selected_dataset):
        """
        Handle file button clicks to load and display shape info.

        Parameters:
        - n_clicks_list: list of int, click counts for each file button
        - avg_filter: str, average filter option ('none', 'avg_faces', 'avg_vertices')
        - selected_category: str, selected category filter ('all' or specific category)
        - sort_field: str, field to sort by ('category', 'num_vertices', 'num_faces')
        - sort_order: str, sort order ('asc' or 'desc')
        - selected_dataset: str, currently selected dataset from dropdown

        Returns:
        - shape_info: HTML component with shape metadata or error message
        - selected_file_idx: int or None, index of the selected file or None to clear selection
        """
        
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update

        trig = ctx.triggered[0]
        prop_id = trig['prop_id']

        # If triggered by file button click
        if 'file-btn' in prop_id:
            if (trig.get('value') or 0) <= 0:
                return no_update, no_update
            try:
                comp_id = json.loads(prop_id.split('.')[0])
                file_idx = comp_id['index']
            except Exception:
                return no_update, no_update

            # Rebuild file_df for current filters
            if selected_dataset is None or selected_dataset == "":
                selected_dataset = 'Data'
            file_df_local = get_file_tree(selected_dataset)
            # Merge analysis CSV columns using cache
            file_df_local = merge_analysis_data(file_df_local, selected_dataset)
            df = file_df_local if selected_category == 'all' else file_df_local[file_df_local['category'] == selected_category]
            ascending = True if sort_order == 'asc' else False
            df = df.copy()
            if sort_field == 'category':
                df = df.sort_values(by=['category', 'filename'], ascending=ascending)
            elif sort_field in ['num_vertices', 'num_faces']:
                # Check if required columns exist for sorting
                if sort_field not in df.columns:
                    error_info = html.Div([
                        html.H4("⚠️ Sorting Not Available", style={'color': '#f39c12', 'marginBottom': '15px'}),
                        html.Div([
                            html.Strong("Missing Data: "), 
                            f"Cannot sort by {sort_field} - analysis data not available for this dataset."
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Strong("Suggestion: "), 
                            "Try sorting by Category instead, or ensure the analysis CSV file exists."
                        ], style={'color': '#7f8c8d'})
                    ])
                    return error_info, None
                df[sort_field] = df[sort_field].fillna(0)
                df = df.sort_values(by=sort_field, ascending=ascending)
            # Apply average filtering after sorting
            if avg_filter == 'avg_faces' and 'num_faces' in df.columns and not df.empty:
                valid = df['num_faces'].dropna()
                if not valid.empty:
                    avg_f = valid.mean()
                    idx = (df['num_faces'] - avg_f).abs().idxmin()
                    if idx in df.index:
                        df = df.loc[[idx]].reset_index(drop=True)
                    else:
                        return no_update, no_update
                else:
                    return no_update, no_update
            elif avg_filter == 'avg_vertices' and 'num_vertices' in df.columns and not df.empty:
                valid = df['num_vertices'].dropna()
                if not valid.empty:
                    avg_v = valid.mean()
                    idx = (df['num_vertices'] - avg_v).abs().idxmin()
                    if idx in df.index:
                        df = df.loc[[idx]].reset_index(drop=True)
                    else:
                        return no_update, no_update
                else:
                    return no_update, no_update
            df = df.reset_index(drop=True)
            if file_idx >= len(df):
                return no_update, no_update
            row = df.iloc[file_idx]
            try:
                mesh = ShapeMesh.from_file_row(row)
                info = mesh.get_card_header_html()
                return info, file_idx
            except Exception as e:
                err = html.Div([
                    html.H4("❌ Error Loading File", style={'color': '#e74c3c', 'marginBottom': '15px'}),
                    html.Div([html.Strong("📄 File: "), row['filepath']], style={'marginBottom': '8px'}),
                    html.Div([html.Strong("⚠️ Error: "), str(e)], style={'color': '#e74c3c'})
                ])
                return err, file_idx
        else:
            # If triggered by filter/sort/dataset change, clear selection
            empty_info = html.P("🔍 Select a 3D shape from the list to view details", className="shape-info-hint")
            return empty_info, None

    # 4) 3D viewer update
    @app.callback(
        Output('3d-plot', 'figure'),
        [Input('display-options', 'value'),
         Input('selected-file-store', 'data'),
         Input('color-selector', 'value'),
         Input('selected-dataset-store', 'data')],
        [State('average-filter', 'value'),
         State('category-filter', 'value'),
         State('sort-field', 'value'),
         State('sort-order', 'value'),
         Input('3d-plot', 'figure')],
        prevent_initial_call=True
    )
    def update_plot(display_options, 
                    selected_file_idx, 
                    mesh_color,                      
                    selected_dataset,
                    avg_filter, 
                    selected_category, 
                    sort_field, 
                    sort_order,
                    current_fig):
        
        """
        Update the 3D plot based on user selections and current figure state.

        Parameters:
        - display_options: list of str, display options selected (e.g., 'wireframe', 'smooth_shading')
        - selected_file_idx: int or None, index of the selected file from the file list
        - mesh_color: str, color selected for the mesh
        - selected_dataset: str, currently selected dataset from dropdown
        - avg_filter: str, average filter option ('none', 'avg_faces', 'avg_vertices')
        - selected_category: str, selected category filter ('all' or specific category)
        - sort_field: str, field to sort by ('category', 'num_vertices', 'num_faces')
        - sort_order: str, sort order ('asc' or 'desc')
        - current_fig: dict, current figure state of the 3D plot
        
        Returns:
        - fig: Plotly figure object for the 3D plot
        - If no shape is selected or an error occurs, returns an empty plot with a message.
        """
        smooth_shading = 'smooth_shading' in (display_options or [])
        camera = None
        if current_fig and 'layout' in current_fig and 'scene' in current_fig['layout']:
            camera = current_fig['layout']['scene'].get('camera', None)
        if selected_file_idx is None:
            return create_3d_plot(np.array([]), np.array([]), "Select a shape to view",
                                  mesh_color=mesh_color or 'lightblue')

        if selected_dataset is None or selected_dataset == "":
            selected_dataset = 'Data'
        
        file_df = get_file_tree(selected_dataset)
        # Merge analysis CSV columns using cache
        file_df = merge_analysis_data(file_df, selected_dataset)

        if file_df is None or file_df.empty:
            return create_3d_plot(np.array([]), np.array([]), "No valid shape selected",
                                  mesh_color=mesh_color or 'lightblue')

        df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
        ascending = True if sort_order == 'asc' else False
        df = df.copy()
        if sort_field == 'category':
            df = df.sort_values(by=['category', 'filename'], ascending=ascending)
        elif sort_field in ['num_vertices', 'num_faces']:
            # Check if required columns exist for sorting
            if sort_field not in df.columns:
                return create_3d_plot(np.array([]), np.array([]), 
                                    f"⚠️ Cannot sort by {sort_field} - analysis data not available",
                                    mesh_color=mesh_color or 'lightblue')
            df[sort_field] = df[sort_field].fillna(0)
            df = df.sort_values(by=sort_field, ascending=ascending)
        # Reset index after sorting/filtering for consistent button indexing
        df = df.reset_index(drop=True)
        # Apply average filtering after sorting
        if avg_filter == 'avg_faces' and 'num_faces' in df.columns and not df.empty:
            valid = df['num_faces'].dropna()
            if not valid.empty:
                avg_f = valid.mean()
                idx = (df['num_faces'] - avg_f).abs().idxmin()
                if idx in df.index:
                    df = df.loc[[idx]].reset_index(drop=True)
                else:
                    return create_3d_plot(np.array([]), np.array([]), "Select a shape to view",
                                         mesh_color=mesh_color or 'lightblue')
            else:
                return create_3d_plot(np.array([]), np.array([]), "Select a shape to view",
                                     mesh_color=mesh_color or 'lightblue')
        elif avg_filter == 'avg_vertices' and 'num_vertices' in df.columns and not df.empty:
            valid = df['num_vertices'].dropna()
            if not valid.empty:
                avg_v = valid.mean()
                idx = (df['num_vertices'] - avg_v).abs().idxmin()
                if idx in df.index:
                    df = df.loc[[idx]].reset_index(drop=True)
                else:
                    return create_3d_plot(np.array([]), np.array([]), "Select a shape to view",
                                         mesh_color=mesh_color or 'lightblue')
            else:
                return create_3d_plot(np.array([]), np.array([]), "Select a shape to view",
                                     mesh_color=mesh_color or 'lightblue')
        # Now use the filtered/sorted DataFrame for index lookup
        if selected_file_idx >= len(df):
            return create_3d_plot(np.array([]), np.array([]), "Select a shape to view",
                                  mesh_color=mesh_color or 'lightblue')
        row = df.iloc[selected_file_idx]
        vertices, faces = OBJParser.parse_obj_file(row['filepath'])
        show_wire = 'wireframe' in (display_options or [])
        title = f"{row['category']} - {row['filename']}"


        fig = create_3d_plot(vertices, faces, title, show_wireframe=show_wire,
                              mesh_color=mesh_color or 'lightblue',
                              smooth_shading=smooth_shading)
        if camera:
            fig.update_layout(scene_camera=camera)
        return fig


    # 5) Similar shapes rendering
    @app.callback(
        Output('aux-plots-row', 'children'),
        [Input('find-shapes-button', 'n_clicks'),
        Input('amount-plots-slider', 'value'),
        Input('selected-file-store', 'data'),
        Input('display-options', 'value'),
        Input('color-selector', 'value')],
        prevent_initial_call=True
    )
    def render_or_clear_aux_plots(n_clicks, n_plots, selected_idx, display_opts, mesh_color):
        """
        Render auxiliary plots of similar shapes when the button is clicked.

        Parameters:
        - n_clicks: int, number of times the "Find Similar Shapes" button was clicked
        - n_plots: int, number of similar shapes to display
        - selected_idx: int or None, index of the selected file from the file list
        - display_opts: list of str, display options selected (e.g., 'wireframe', 'smooth_shading')
        - mesh_color: str, color selected for the mesh

        Returns:
        - List of HTML Div elements containing the auxiliary plots or no_update/empty list to clear
        """
        
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Clear the row when a new shape is selected
        if triggered_id == 'selected-file-store':
            return []

        # Only render after button click
        if triggered_id != 'find-shapes-button' or not n_clicks or n_clicks <= 0 or selected_idx is None:
            return no_update

        # For now, this function doesn't have access to file_df context 
        # This is a placeholder implementation - similar shapes functionality
        # would need significant refactoring to work with the new architecture 
        
        # Create a simple placeholder mesh for demonstration
        try:
            # Create dummy vertices for now (this would need proper similar shape logic)
            vertices = np.random.rand(100, 3) * 2 - 1  # Random vertices from -1 to 1
            faces = np.array([[0, 1, 2], [1, 2, 3]])  # Dummy faces
        except Exception:
            return []

        show_wire = 'wireframe' in (display_opts or [])
        smooth_shading = 'smooth_shading' in (display_opts or [])
        total = int(n_plots or 5)
        title = "Similar Shape"

        # Render cards with independent plot objects
        cards = []
        for i in range(total):
            # Deep copy vertices and faces to ensure independence
            v_copy = np.copy(vertices)
            f_copy = np.copy(faces)
            card_title = f"{title} (Aux {i+1})"
            fig = create_3d_plot(v_copy, f_copy, card_title, show_wireframe=show_wire,
                                mesh_color=mesh_color or 'lightblue',
                                smooth_shading=smooth_shading)

            # Create a simple header for aux plots
            header = html.Div([
                html.Div([
                    html.Span("🔍 ", className="shape-info-icon"), 
                    html.Strong(f"Similar Shape {i+1}")
                ], className="shape-info-prop")
            ], className="shape-info-header")

            card = html.Div([
                header,
                dcc.Graph(figure=fig, 
                          className='three-d-plot')
            ], style={
                'minWidth': '360px',
                'height': '200px',
                'backgroundColor': '#fff',
                'border': '1px solid #e1e1e1',
                'borderRadius': '8px',
                'boxShadow': '0 1px 4px rgba(0,0,0,0.06)',
                'padding': '6px'
            })

            cards.append(card)

        return cards
    
     # Store current dataset in dcc.Store
    @app.callback(
        Output('selected-dataset-store', 'data'),
        Input('dataset-selector', 'value'),
        State('selected-dataset-store', 'data')
    )
    def update_selected_dataset(selected_dataset, current_dataset):
        """
        Update the selected dataset store when the dropdown changes.

        Parameters:
        - selected_dataset: str, newly selected dataset from dropdown
        - current_dataset: str, currently stored dataset

        Returns:
        - str, updated dataset value
        """
        if selected_dataset and selected_dataset != current_dataset:
            return selected_dataset
        return current_dataset

    # Update category filter options when dataset changes
    @app.callback(
        [Output('category-filter', 'options'), Output('category-filter', 'value')],
        Input('selected-dataset-store', 'data'),
        State('category-filter', 'value')
    )
    def update_category_options(selected_dataset, current_category):
        """
        Update the category filter options based on the selected dataset.

        Parameters:
        - selected_dataset: str, currently selected dataset from dropdown
        - current_category: str, currently selected category filter

        Returns:
        - options: list of dict, updated options for the category filter dropdown
        - value: str, updated selected value for the category filter dropdown
        """
        if not selected_dataset:
            selected_dataset = 'Data'
        
        try:
            file_df = get_file_tree(selected_dataset)
            if file_df.empty:
                options = [{'label': 'All Categories', 'value': 'all'}]
                return options, 'all'
            
            options = [{'label': 'All Categories', 'value': 'all'}] + \
                     [{'label': cat, 'value': cat} for cat in sorted(file_df['category'].unique())]
            
            # Check if current category is still valid
            valid_values = [opt['value'] for opt in options]
            if current_category in valid_values:
                return options, current_category
            else:
                return options, 'all'
        except Exception as e:
            print(f"[DEBUG] Error updating category options: {e}")
            return [{'label': 'All Categories', 'value': 'all'}], 'all'

