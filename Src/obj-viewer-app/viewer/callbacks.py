
from dash import dcc, html, Input, Output, no_update, State, callback_context
import dash
import numpy as np
import os
import pandas as pd
import json
from core.obj_parser import OBJParser
from core.plotting import create_3d_plot
import plotly.graph_objects as go

def register_callbacks(app: dash.Dash, file_df, dataset_options, default_dataset):
    # Combined callback: reset file index on dataset change, set index on file button click
    @app.callback(
        [Output('shape-info', 'children'), Output('selected-file-store', 'data')],
        [Input({'type': 'file-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
         Input('selected-dataset-store', 'data')],
        [State('average-filter', 'value'),
         State('category-filter', 'value'),
         State('sort-field', 'value'),
         State('sort-order', 'value')],
        prevent_initial_call=True
    )
    def update_file_selection_or_reset(n_clicks_list, selected_dataset, avg_filter, selected_category, sort_field, sort_order):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # If dataset changed, reset file selection
        if triggered_id == 'selected-dataset-store':
            return no_update, None
        # If file button clicked, select file
        trig = ctx.triggered[0]
        if (trig.get('value') or 0) <= 0 or 'file-btn' not in trig['prop_id']:
            return no_update, no_update
        try:
            comp_id = json.loads(trig['prop_id'].split('.')[0])
            file_idx = comp_id['index']
        except Exception:
            return no_update, no_update
        # Reproduce the same filtering and sorting logic as update_file_list
        from core.file_index import get_file_tree
        if not selected_dataset:
            selected_dataset = 'Data'
        file_df = get_file_tree(selected_dataset)
        # Merge analysis CSV columns
        if selected_dataset == 'Data':
            analysis_path = 'Preprocessing/analysis_results.csv'
        elif selected_dataset == 'Data_sampled':
            analysis_path = 'Preprocessing/analysis_results_sampled.csv'
        elif selected_dataset == 'Data_sampled_resampled':
            analysis_path = 'Preprocessing/analysis_results_sampled_resampled.csv'
        else:
            analysis_path = None
        if analysis_path:
            try:
                analysis_df = pd.read_csv(analysis_path)
                analysis_df = analysis_df.rename(columns={
                    'class': 'category',
                    'shape_file': 'filename'
                })
                file_df = pd.merge(file_df, analysis_df[['category', 'filename', 'num_vertices', 'num_faces']],
                                   on=['category', 'filename'], how='left')
            except Exception as e:
                print(f"[DEBUG] Could not merge analysis CSV: {e}")
        df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
        ascending = True if sort_order == 'asc' else False
        df = df.copy()
        if sort_field == 'category':
            df = df.sort_values(by=['category', 'filename'], ascending=ascending)
        elif sort_field in ['num_vertices', 'num_faces']:
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
        # Now use the filtered/sorted DataFrame for index lookup
        if file_idx >= len(df):
            return no_update, no_update
        row = df.iloc[file_idx]
        vertices, faces = OBJParser.parse_obj_file(row['filepath'])
        try:
            if vertices.size > 0:
                minc = vertices.min(axis=0)
                maxc = vertices.max(axis=0)
                dims = maxc - minc
            else:
                dims = [0, 0, 0]
            quality = "Good" if (len(vertices) > 100 and len(faces) > 50) else "Low Resolution"
            info = html.Div([
                html.H4(["✅ ", row['filename']], style={
                    'marginBottom': '15px', 'color': '#27ae60',
                    'borderBottom': '2px solid #27ae60', 'paddingBottom': '8px'
                }),
                html.Div([
                    html.Div([html.Strong("📁 Category: "), row['category']], style={'marginBottom': '8px'}),
                    html.Div([html.Strong("💾 File Size: "), f"{row['size']:,} bytes"], style={'marginBottom': '8px'}),
                    html.Div([html.Strong("🔺 Vertices: "), f"{len(vertices):,}"], style={'marginBottom': '8px'}),
                    html.Div([html.Strong("🔷 Faces: "), f"{len(faces):,}"], style={'marginBottom': '8px'}),
                    html.Div([html.Strong("📐 Dimensions: "),
                              f"X: {dims[0]:.2f}, Y: {dims[1]:.2f}, Z: {dims[2]:.2f}"],
                             style={'marginBottom': '8px'}),
                    html.Div([html.Strong("🎯 Quality: "), quality], style={'marginBottom': '8px'}),
                ])
            ])
            return info, file_idx
        except Exception as e:
            err = html.Div([
                html.H4("❌ Error Loading File", style={'color': '#e74c3c', 'marginBottom': '15px'}),
                html.Div([html.Strong("📄 File: "), row['filepath']], style={'marginBottom': '8px'}),
                html.Div([html.Strong("⚠️ Error: "), str(e)], style={'color': '#e74c3c'})
            ])
            return err, file_idx
    # Store current dataset in dcc.Store
    @app.callback(
        Output('selected-dataset-store', 'data'),
        Input('dataset-selector', 'value'),
        State('selected-dataset-store', 'data')
    )
    def update_selected_dataset(selected_dataset, current_dataset):
        if selected_dataset and selected_dataset != current_dataset:
            return selected_dataset
        return current_dataset

    @app.callback(
        Output('file-list', 'children'),
        [Input('average-filter', 'value'),
         Input('category-filter', 'value'),
         Input('sort-field', 'value'),
         Input('sort-order', 'value'),
         Input('selected-dataset-store', 'data')]
    )
    def update_file_list(avg_filter, selected_category, sort_field, sort_order, selected_dataset):
        # Debug: log avg_filter value every time callback runs
        print(f"[DEBUG] Callback triggered. avg_filter={avg_filter}")
        from core.file_index import get_file_tree
        import pandas as pd
        if not selected_dataset:
            selected_dataset = 'Data'
        file_df = get_file_tree(selected_dataset)
        # Merge analysis CSV columns
        if selected_dataset == 'Data':
            analysis_path = 'Preprocessing/analysis_results.csv'
        elif selected_dataset == 'Data_sampled':
            analysis_path = 'Preprocessing/analysis_results_sampled.csv'
        elif selected_dataset == 'Data_sampled_resampled':
            analysis_path = 'Preprocessing/analysis_results_sampled_resampled.csv'
        else:
            analysis_path = None
        if analysis_path:
            try:
                analysis_df = pd.read_csv(analysis_path)
                # Rename columns to match file_df
                analysis_df = analysis_df.rename(columns={
                    'class': 'category',
                    'shape_file': 'filename'
                })
                # Merge on category and filename
                file_df = pd.merge(file_df, analysis_df[['category', 'filename', 'num_vertices', 'num_faces']],
                                   on=['category', 'filename'], how='left')
            except Exception as e:
                print(f"[DEBUG] Could not merge analysis CSV: {e}")
        if file_df.empty:
            return [html.P("❌ No files found in Data directory",
                           style={'color': 'red', 'textAlign': 'center'})]
        df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]

        ascending = True if sort_order == 'asc' else False
        df = df.copy()
        if sort_field == 'category':
            df = df.sort_values(by=['category', 'filename'], ascending=ascending)
        elif sort_field in ['num_vertices', 'num_faces']:
            df[sort_field] = df[sort_field].fillna(0)
            df = df.sort_values(by=sort_field, ascending=ascending)

        
        print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
        print(f"[DEBUG] DataFrame length before avg filter: {len(df)}")

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
            print(f"[DEBUG] Rows after avg_faces filter: {len(df)}")
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
            print(f"[DEBUG] Rows after avg_vertices filter: {len(df)}")

        buttons = []
        for idx, row in df.iterrows():
            btn = html.Button(
                html.Div([
                    html.Strong(f"📁 {row['category']}", className="category-text"),
                    html.Br(),
                    html.Span(f"📄 {row['filename']}", className="filename-text")
                ]),
                id={'type': 'file-btn', 'index': int(idx)},
                className='file-button',
                n_clicks=0,
                **{'data-file-index': int(idx)}
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
         State('sort-order', 'value')],
        prevent_initial_call=True
    )
    def update_plot(display_options, selected_file_idx, mesh_color, selected_dataset, avg_filter, selected_category, sort_field, sort_order):
        if selected_file_idx is None:
            return create_3d_plot(np.array([]), np.array([]), "Select a shape to view",
                                  mesh_color=mesh_color or 'lightblue')
        if not selected_dataset:
            selected_dataset = 'Data'
        from core.file_index import get_file_tree
        import pandas as pd
        file_df = get_file_tree(selected_dataset)
        # Merge analysis CSV columns
        if selected_dataset == 'Data':
            analysis_path = 'Preprocessing/analysis_results.csv'
        elif selected_dataset == 'Data_sampled':
            analysis_path = 'Preprocessing/analysis_results_sampled.csv'
        elif selected_dataset == 'Data_sampled_resampled':
            analysis_path = 'Preprocessing/analysis_results_sampled_resampled.csv'
        else:
            analysis_path = None
        if analysis_path:
            try:
                analysis_df = pd.read_csv(analysis_path)
                analysis_df = analysis_df.rename(columns={
                    'class': 'category',
                    'shape_file': 'filename'
                })
                file_df = pd.merge(file_df, analysis_df[['category', 'filename', 'num_vertices', 'num_faces']],
                                   on=['category', 'filename'], how='left')
            except Exception as e:
                print(f"[DEBUG] Could not merge analysis CSV: {e}")
        df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
        ascending = True if sort_order == 'asc' else False
        df = df.copy()
        if sort_field == 'category':
            df = df.sort_values(by=['category', 'filename'], ascending=ascending)
        elif sort_field in ['num_vertices', 'num_faces']:
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
        return create_3d_plot(vertices, faces, title, show_wireframe=show_wire,
                              mesh_color=mesh_color or 'lightblue')

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

        # Load selected shape
        try:
            row = file_df.iloc[selected_idx]
            filepath = row['filepath']
        except Exception:
            return no_update

        try:
            vertices, faces = OBJParser.parse_obj_file(filepath)
        except Exception:
            return []

        show_wire = 'wireframe' in (display_opts or [])
        total = int(n_plots or 5)
        title = f"{row['category']} - {row['filename']}"

        
        if vertices.size > 0:
            minc = vertices.min(axis=0)
            maxc = vertices.max(axis=0)
            dims = maxc - minc
        else:
            dims = np.array([0.0, 0.0, 0.0])

        quality = "Good" if (len(vertices) > 100 and len(faces) > 50) else "Low Resolution"

        def _num(n):
            return f"{int(n):,}"

        def _bytes(b):
            try:
                b = int(b)
            except Exception:
                return "-"
            units = ["B","KB","MB","GB","TB"]
            i = 0
            x = float(b)
            while x >= 1024 and i < len(units)-1:
                x /= 1024.0
                i += 1
            return f"{x:.1f} {units[i]}"

        # Render cards ()
        cards = []
        for i in range(total):
            fig = create_3d_plot(vertices, faces, title, show_wireframe=show_wire,
                                mesh_color=mesh_color or 'lightblue')

            header = html.Div([
                html.Div([
                    html.Span("📁 ", style={'marginRight': '4px'}), html.Strong("Category: "),
                    html.Span(row['category'])
                ], style={'marginRight': '12px'}),
                html.Div([
                    html.Span("💾 ", style={'marginRight': '4px'}), html.Strong("Size: "),
                    html.Span(_bytes(row.get('size', 0)))
                ], style={'marginRight': '12px'}),
                html.Div([
                    html.Span("🔺 ", style={'marginRight': '4px'}), html.Strong("Vertices: "),
                    html.Span(_num(len(vertices)))
                ], style={'marginRight': '12px'}),
                html.Div([
                    html.Span("🔷 ", style={'marginRight': '4px'}), html.Strong("Faces: "),
                    html.Span(_num(len(faces)))
                ], style={'marginRight': '12px'}),
                html.Div([
                    html.Span("📐 ", style={'marginRight': '4px'}), html.Strong("Dims: "),
                    html.Span(f"X {dims[0]:.2f} · Y {dims[1]:.2f} · Z {dims[2]:.2f}")
                ], style={'marginRight': '12px'}),
                html.Div([
                    html.Span("🎯 ", style={'marginRight': '4px'}), html.Strong("Quality: "),
                    html.Span(quality)
                ]),
            ], style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'rowGap': '4px',
                'columnGap': '8px',
                'fontSize': '12px',
                'color': '#2c3e50',
                'padding': '6px 6px 4px 6px',
                'borderBottom': '1px solid #eee',
                'marginBottom': '6px',
                'height': '50px'
            })

            card = html.Div([
                header,
                dcc.Graph(figure=fig, style={'height': '220px', 'width': '360px'})
            ], style={
                'minWidth': '360px',
                'height': '350px',
                'backgroundColor': '#fff',
                'border': '1px solid #e1e1e1',
                'borderRadius': '8px',
                'boxShadow': '0 1px 4px rgba(0,0,0,0.06)',
                'padding': '6px'
            })

            cards.append(card)

        return cards