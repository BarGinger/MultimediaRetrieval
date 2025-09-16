
from dash import dcc, html, Input, Output, no_update
import dash
import numpy as np
import os
import pandas as pd
from core.obj_parser import OBJParser
from core.plotting import create_3d_plot
import plotly.graph_objects as go

def register_callbacks(app: dash.Dash, file_df, USE_SAMPLED_DATASET):
    # Select and merge CSV for statistics
    suffix = '_sampled' if USE_SAMPLED_DATASET else ''
    csv_filename = f'analysis_results{suffix}.csv'
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Preprocessing', csv_filename)
    csv_path = os.path.abspath(csv_path)

    print(f"Loaded {len(file_df)} files from {'Data_sampled' if USE_SAMPLED_DATASET else 'Data'} (USE_SAMPLED_DATASET={USE_SAMPLED_DATASET})")

    if os.path.exists(csv_path):
        analysis_df = pd.read_csv(csv_path)
        # Merge on filename and category/class
        if 'class' in analysis_df.columns:
            merged_df = pd.merge(file_df, analysis_df, left_on=['filename', 'category'], right_on=['shape_file', 'class'], how='left')
            # Fill missing values with 0 for sorting
            merged_df['num_vertices'] = merged_df['num_vertices'].fillna(0)
            merged_df['num_faces'] = merged_df['num_faces'].fillna(0)
            file_df = merged_df
        else:
            print(f'{csv_filename} missing required columns.')
    else:
        print(f'{csv_filename} not found, sorting by vertex/face count will not work.')




    # 1) File list render with sorting
    @app.callback(
        Output('file-list', 'children'),
        [Input('average-filter', 'value'),
         Input('category-filter', 'value'),
         Input('sort-field', 'value'),
         Input('sort-order', 'value')]
    )
    def update_file_list(avg_filter, selected_category, sort_field, sort_order):
        if file_df.empty:
            return [html.P("❌ No files found in Data directory",
                           style={'color': 'red', 'textAlign': 'center'})]

        # Filter for average shape if requested, within selected category
        df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
        if avg_filter == 'avg_faces' and 'num_faces' in df.columns and not df.empty:
            valid = df['num_faces'].dropna()
            if not valid.empty:
                avg_f = valid.mean()
                idx = (df['num_faces'] - avg_f).abs().idxmin()
                if idx in df.index:
                    df = df.loc[[idx]]
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
                    df = df.loc[[idx]]
                else:
                    return [html.P("❌ No valid shapes for average by vertices", style={'color': 'orange', 'textAlign': 'center'})]
            else:
                return [html.P("❌ No valid shapes for average by vertices", style={'color': 'orange', 'textAlign': 'center'})]

        if df.empty:
            return [html.P("❌ No files found for selection", style={'color': 'orange', 'textAlign': 'center'})]

        # Sorting logic
        ascending = True if sort_order == 'asc' else False
        df = df.copy()
        if sort_field == 'category':
            df = df.sort_values(by=['category', 'filename'], ascending=ascending)
        elif sort_field in ['num_vertices', 'num_faces']:
            df[sort_field] = df[sort_field].fillna(0)
            df = df.sort_values(by=sort_field, ascending=ascending)

        buttons = []
        for idx, row in df.iterrows():
            buttons.append(html.Button(
                html.Div([
                    html.Strong(f"📁 {row['category']}", className="category-text"),
                    html.Br(),
                    html.Span(f"📄 {row['filename']}", className="filename-text")
                ]),
                id={'type': 'file-btn', 'index': idx},
                className='file-button',
                n_clicks=0,
                **{'data-file-index': idx}
            ))
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
        [Input({'type': 'file-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
        prevent_initial_call=True
    )
    def select_file(n_clicks_list):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update

        trig = ctx.triggered[0]
        if (trig.get('value') or 0) <= 0 or 'file-btn' not in trig['prop_id']:
            return no_update, no_update

        import json
        try:
            comp_id = json.loads(trig['prop_id'].split('.')[0])
            file_idx = comp_id['index']
        except Exception:
            return no_update, no_update

        if file_idx >= len(file_df):
            return no_update, no_update

        row = file_df.iloc[file_idx]
        filepath = row['filepath']

        try:
            vertices, faces = OBJParser.parse_obj_file(filepath)
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
                html.Div([html.Strong("📄 File: "), filepath], style={'marginBottom': '8px'}),
                html.Div([html.Strong("⚠️ Error: "), str(e)], style={'color': '#e74c3c'})
            ])
            return err, file_idx

    # 4) 3D viewer update
    @app.callback(
        Output('3d-plot', 'figure'),
        [Input('display-options', 'value'),
         Input('selected-file-store', 'data'),
         Input('color-selector', 'value')],
        prevent_initial_call=True
    )
    def update_plot(display_options, selected_file_idx, mesh_color):
        if selected_file_idx is None:
            return create_3d_plot(np.array([]), np.array([]), "Select a shape to view",
                                  mesh_color=mesh_color or 'lightblue')

        row = file_df.iloc[selected_file_idx]
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