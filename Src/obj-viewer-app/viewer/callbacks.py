
from dash import dcc, html, Input, Output, no_update, State, callback_context
import dash
import numpy as np
import os
import pandas as pd
import json
import uuid
import time
from core.obj_parser import OBJParser
from core.plotting import create_3d_plot
import plotly.graph_objects as go
from core.file_index import get_file_tree, get_step_file_path, get_step_display_info, get_available_steps
from core.analysis_cache import merge_analysis_data, get_analysis_data
from core.dataset_cache import get_cached_dataset_data, get_available_datasets, preload_datasets
from core.shapeMesh import ShapeMesh


def create_toast_data(message, toast_type="info", icon="ℹ️"):
    """Create toast data for store"""
    import random
    return {
        "message": message,
        "type": toast_type,
        "icon": icon,
        "timestamp": time.time(),
        "id": uuid.uuid4().hex[:8],
        "random": random.randint(1, 1000000)  # Extra randomness to force updates
    }


def register_callbacks(app: dash.Dash, file_df, dataset_options, default_dataset):

    # Sort order button toggle - SAME PATTERN AS LOADING MESSAGE
    @app.callback(
        [Output('sort-order', 'children'),
         Output('sort-order', 'title'),
         Output('sort-order', 'data-order'),
         Output('toast-store', 'data', allow_duplicate=True)],
        [Input('sort-order', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_sort_order(n_clicks):
        """Toggle between ascending and descending sort order"""
        print(f"🔄 Sort button clicked! n_clicks: {n_clicks}")  # Debug
        
        if n_clicks is None:
            n_clicks = 0
        
        # Even clicks = ascending, odd clicks = descending
        if n_clicks % 2 == 0:
            print(f"✅ Creating ascending sort")  # Debug
            toast_data = create_toast_data("Sort order changed to Ascending", "info", "↑")
            return "↑", "Sort Order: Ascending (click to change to Descending)", "asc", toast_data
        else:
            print(f"✅ Creating descending sort")  # Debug
            toast_data = create_toast_data("Sort order changed to Descending", "info", "↓")
            return "↓", "Sort Order: Descending (click to change to Ascending)", "desc", toast_data

    # Show toast for filename filter changes
    app.clientside_callback(
        """
        function(filename_filter) {
            if (!filename_filter || filename_filter.trim() === '') {
                return window.dash_clientside.no_update;
            }
            
            // Always hide first, then show to ensure animation works
            const toastBar = document.getElementById('toast-message-bar');
            if (toastBar) {
                toastBar.style.display = 'none';
            }
            
            // Force reflow then show
            setTimeout(function() {
                const toastBar = document.getElementById('toast-message-bar');
                if (toastBar) {
                    document.getElementById('toast-icon').innerHTML = '🔍';
                    document.getElementById('toast-message').innerHTML = 'Filename filter applied: ' + filename_filter;
                    toastBar.style.display = 'block';
                    
                    // Auto-hide after 3 seconds
                    setTimeout(function() {
                        if (toastBar) {
                            toastBar.style.display = 'none';
                        }
                    }, 3000);
                }
            }, 10);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('dummy-div', 'children', allow_duplicate=True),  # Dummy output since we manipulate DOM directly
        Input('filename-filter', 'value'),
        prevent_initial_call=True
    )

    # Show toast for vertices filter changes
    app.clientside_callback(
        """
        function(vertices_op, vertices_val) {
            try {
                if (!vertices_val || vertices_val === '') {
                    return window.dash_clientside.no_update;
                }
                
                const toastBar = document.getElementById('toast-message-bar');
                if (toastBar) {
                    toastBar.style.display = 'none';
                }
                
                setTimeout(function() {
                    const toastBar = document.getElementById('toast-message-bar');
                    const toastIcon = document.getElementById('toast-icon');
                    const toastMessage = document.getElementById('toast-message');
                    
                    if (toastBar && toastIcon && toastMessage) {
                        toastIcon.innerHTML = '📊';
                        const opText = vertices_op === 'eq' ? 'Equal to' : vertices_op === 'gt' ? 'Greater than' : 'Less than';
                        toastMessage.innerHTML = 'Vertices filter: ' + opText + ' ' + vertices_val;
                        toastBar.style.display = 'block';
                        
                        setTimeout(function() {
                            if (toastBar) {
                                toastBar.style.display = 'none';
                            }
                        }, 2000);
                    }
                }, 10);
                
                return window.dash_clientside.no_update;
            } catch (error) {
                console.error('Error in vertices filter toast:', error);
                return window.dash_clientside.no_update;
            }
        }
        """,
        Output('toast-message-bar', 'id', allow_duplicate=True),
        [Input('vertices-operator', 'value'),
         Input('vertices-value', 'value')],
        prevent_initial_call=True
    )

    # Show toast for faces filter changes
    app.clientside_callback(
        """
        function(faces_op, faces_val) {
            try {
                if (!faces_val || faces_val === '') {
                    return window.dash_clientside.no_update;
                }
                
                const toastBar = document.getElementById('toast-message-bar');
                if (toastBar) {
                    toastBar.style.display = 'none';
                }
                
                setTimeout(function() {
                    const toastBar = document.getElementById('toast-message-bar');
                    const toastIcon = document.getElementById('toast-icon');
                    const toastMessage = document.getElementById('toast-message');
                    
                    if (toastBar && toastIcon && toastMessage) {
                        toastIcon.innerHTML = '🔷';
                        const opText = faces_op === 'eq' ? 'Equal to' : faces_op === 'gt' ? 'Greater than' : 'Less than';
                        toastMessage.innerHTML = 'Faces filter: ' + opText + ' ' + faces_val;
                        toastBar.style.display = 'block';
                        
                        setTimeout(function() {
                            if (toastBar) {
                                toastBar.style.display = 'none';
                            }
                        }, 2000);
                    }
                }, 10);
                
                return window.dash_clientside.no_update;
            } catch (error) {
                console.error('Error in faces filter toast:', error);
                return window.dash_clientside.no_update;
            }
        }
        """,
        Output('toast-message-bar', 'id', allow_duplicate=True),
        [Input('faces-operator', 'value'),
         Input('faces-value', 'value')],
        prevent_initial_call=True
    )

    # Clear filters button callback
    @app.callback(
        [Output('category-filter', 'value', allow_duplicate=True),
         Output('filename-filter', 'value', allow_duplicate=True), 
         Output('vertices-operator', 'value', allow_duplicate=True),
         Output('vertices-value', 'value', allow_duplicate=True),
         Output('faces-operator', 'value', allow_duplicate=True),
         Output('faces-value', 'value', allow_duplicate=True),
         Output('sort-field', 'value', allow_duplicate=True)],
        Input('clear-filters-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_filters(n_clicks):
        """Reset all filters to their default values"""
        if n_clicks and n_clicks > 0:
            return 'all', '', 'gt', '', 'gt', '', 'category'
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Show toast for clear filters button
    app.clientside_callback(
        """
        function(n_clicks) {
            try {
                if (!n_clicks || n_clicks === 0) {
                    return window.dash_clientside.no_update;
                }
                
                const toastBar = document.getElementById('toast-message-bar');
                if (toastBar) {
                    toastBar.style.display = 'none';
                }
                
                setTimeout(function() {
                    const toastBar = document.getElementById('toast-message-bar');
                    const toastIcon = document.getElementById('toast-icon');
                    const toastMessage = document.getElementById('toast-message');
                    
                    if (toastBar && toastIcon && toastMessage) {
                        toastIcon.innerHTML = '🧹';
                        toastMessage.innerHTML = 'All filters have been cleared';
                        toastBar.style.display = 'block';
                        
                        setTimeout(function() {
                            if (toastBar) {
                                toastBar.style.display = 'none';
                            }
                        }, 2000);
                    }
                }, 10);
                
                return window.dash_clientside.no_update;
            } catch (error) {
                console.error('Error in clear filters toast:', error);
                return window.dash_clientside.no_update;
            }
        }
        """,
        Output('toast-message-bar', 'id', allow_duplicate=True),
        Input('clear-filters-btn', 'n_clicks'),
        prevent_initial_call=True
    )

    # Show toast for average vertices button
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            
            // Always hide first, then show to ensure animation works
            const toastBar = document.getElementById('toast-message-bar');
            if (toastBar) {
                toastBar.style.display = 'none';
            }
            
            // Force reflow then show
            setTimeout(function() {
                const toastBar = document.getElementById('toast-message-bar');
                if (toastBar) {
                    document.getElementById('toast-icon').innerHTML = '📊';
                    document.getElementById('toast-message').innerHTML = 'Scrolling to Average Vertices shape';
                    toastBar.style.display = 'block';
                    
                    // Auto-hide after 3 seconds
                    setTimeout(function() {
                        if (toastBar) {
                            toastBar.style.display = 'none';
                        }
                    }, 3000);
                }
            }, 10);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('toast-message-bar', 'id', allow_duplicate=True),
        Input('avg-vertices-btn', 'n_clicks'),
        prevent_initial_call=True
    )

    # Show toast for average faces button  
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            
            // Always hide first, then show to ensure animation works
            const toastBar = document.getElementById('toast-message-bar');
            if (toastBar) {
                toastBar.style.display = 'none';
            }
            
            // Force reflow then show
            setTimeout(function() {
                const toastBar = document.getElementById('toast-message-bar');
                if (toastBar) {
                    document.getElementById('toast-icon').innerHTML = '🔷';
                    document.getElementById('toast-message').innerHTML = 'Scrolling to Average Faces shape';
                    toastBar.style.display = 'block';
                    
                    // Auto-hide after 3 seconds
                    setTimeout(function() {
                        if (toastBar) {
                            toastBar.style.display = 'none';
                        }
                    }, 3000);
                }
            }, 10);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('toast-message-bar', 'id', allow_duplicate=True),
        Input('avg-faces-btn', 'n_clicks'),
        prevent_initial_call=True
    )

    # Show toast for shape loading when file button is clicked
    app.clientside_callback(
        """
        function(n_clicks_list) {
            const ctx = window.dash_clientside.callback_context;
            if (!ctx.triggered.length) {
                return window.dash_clientside.no_update;
            }
            
            const trigger = ctx.triggered[0];
            const propId = trigger.prop_id;
            
            // Check if it's a file button click
            if (propId.includes('file-btn') && trigger.value > 0) {
                // Always hide first, then show to ensure animation works
                const toastBar = document.getElementById('toast-message-bar');
                if (toastBar) {
                    toastBar.style.display = 'none';
                }
                
                // Force reflow then show
                setTimeout(function() {
                    const toastBar = document.getElementById('toast-message-bar');
                    const toastIcon = document.getElementById('toast-icon');
                    const toastMessage = document.getElementById('toast-message');
                    
                    if (toastBar && toastIcon && toastMessage) {
                        toastIcon.innerHTML = '🔄';
                        toastMessage.innerHTML = 'Loading selected shape...';
                        toastBar.style.display = 'block';
                        
                        // Auto-hide after 4 seconds for loading toast
                        setTimeout(function() {
                            if (toastBar) {
                                toastBar.style.display = 'none';
                            }
                        }, 4000);
                    }
                }, 10);
            }
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('toast-message-bar', 'id', allow_duplicate=True),
        Input({'type': 'file-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
        prevent_initial_call=True
    )

    # Show global loading indicator when file button is clicked
    app.clientside_callback(
        """
        function(n_clicks_list) {
            const ctx = window.dash_clientside.callback_context;
            if (!ctx.triggered.length) {
                return window.dash_clientside.no_update;
            }
            
            const trigger = ctx.triggered[0];
            const propId = trigger.prop_id;
            
            // Check if it's a file button click
            if (propId.includes('file-btn') && trigger.value > 0) {
                const loadingIndicator = document.getElementById('global-loading-indicator');
                if (loadingIndicator) {
                    loadingIndicator.style.display = 'block';
                }
            }
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('global-loading-indicator', 'id', allow_duplicate=True),
        Input({'type': 'file-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
        prevent_initial_call=True
    )

    # Hide global loading indicator when shape info is updated
    app.clientside_callback(
        """
        function(shape_info) {
            // Don't hide on shape info update - let the 3D plot callback handle it
            return window.dash_clientside.no_update;
        }
        """,
        Output('global-loading-indicator', 'id', allow_duplicate=True),
        Input('shape-info', 'children'),
        prevent_initial_call=True
    )

    # Hide global loading indicator when 3D plot is updated with actual shape data
    app.clientside_callback(
        """
        function(figure) {
            const loadingIndicator = document.getElementById('global-loading-indicator');
            if (loadingIndicator && figure && figure.data && figure.data.length > 0) {
                // Check if the plot has any data traces (more reliable detection)
                const hasData = figure.data.length > 0 && (
                    // Check for mesh3d data
                    figure.data.some(trace => trace.type === 'mesh3d') ||
                    // Check for scatter3d data 
                    figure.data.some(trace => trace.type === 'scatter3d') ||
                    // Check for any trace with x data
                    figure.data.some(trace => trace.x && trace.x.length > 0)
                );
                
                if (hasData) {
                    // Hide immediately when data is detected
                    setTimeout(function() {
                        loadingIndicator.style.display = 'none';
                    }, 100);
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('global-loading-indicator', 'id', allow_duplicate=True),
        Input('3d-plot', 'figure'),
        prevent_initial_call=True
    )

    # Average navigation buttons
    @app.callback(
        [Output('selected-file-store', 'data', allow_duplicate=True),
         Output('shape-info', 'children', allow_duplicate=True),
         Output('toast-store', 'data', allow_duplicate=True)],
        [Input('avg-vertices-btn', 'n_clicks'),
         Input('avg-faces-btn', 'n_clicks')],
        [State('category-filter', 'value'),
         State('filename-filter', 'value'),
         State('vertices-operator', 'value'),
         State('vertices-value', 'value'),
         State('faces-operator', 'value'),
         State('faces-value', 'value'),
         State('sort-field', 'value'),
         State('sort-order', 'data-order'),
         State('selected-dataset-store', 'data')],
        prevent_initial_call=True
    )
    def navigate_to_average(avg_vertices_clicks, avg_faces_clicks, selected_category, filename_filter, vertices_op, vertices_val, faces_op, faces_val, sort_field, sort_order, selected_dataset):
        """Navigate to the item closest to average vertices or faces in the currently displayed list"""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # CRITICAL: Get the current dataset and apply THE SAME FILTERS as the file list
        # This ensures the index we calculate matches what's displayed in the UI
        try:
            file_df = get_cached_dataset_data(selected_dataset)
        except Exception as e:
            print(f"Error loading dataset {selected_dataset}: {e}")
            return no_update, no_update, no_update
        
        # Cached data already includes analysis columns (num_vertices, num_faces)
        print(f"✅ Using cached data for average navigation with {len(file_df)} shapes")
        
        # Verify analysis data is present
        if 'num_vertices' not in file_df.columns or 'num_faces' not in file_df.columns:
            print(f"⚠️ No analysis data in cached dataset {selected_dataset} - cannot find average")
            toast_data = create_toast_data("No analysis data available for average calculation", "warning", "⚠️")
            return no_update, no_update, toast_data
        
        # Apply category filter (same as file list)
        df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
        
        # Apply filename filtering (same as file list)
        if filename_filter and filename_filter.strip() and not df.empty and 'filename' in df.columns:
            try:
                import fnmatch
                pattern = filename_filter.strip()
                mask = df['filename'].apply(lambda x: fnmatch.fnmatch(x.lower(), pattern.lower()))
                df = df[mask]
            except Exception as e:
                print(f"Error applying filename filter '{filename_filter}': {e}")
        
        # Apply vertices filtering (same as file list)
        if vertices_val is not None and vertices_val != '' and 'num_vertices' in df.columns:
            try:
                val = int(vertices_val)
                if vertices_op == 'eq':
                    df = df[df['num_vertices'] == val]
                elif vertices_op == 'gt':
                    df = df[df['num_vertices'] > val]
                elif vertices_op == 'lt':
                    df = df[df['num_vertices'] < val]
            except ValueError:
                pass
        
        # Apply faces filtering (same as file list)
        if faces_val is not None and faces_val != '' and 'num_faces' in df.columns:
            try:
                val = int(faces_val)
                if faces_op == 'eq':
                    df = df[df['num_faces'] == val]
                elif faces_op == 'gt':
                    df = df[df['num_faces'] > val]
                elif faces_op == 'lt':
                    df = df[df['num_faces'] < val]
            except ValueError:
                pass
        
        # Apply sorting (same as file list)
        ascending = True if sort_order == 'asc' else False
        df = df.copy()
        if sort_field == 'category':
            df = df.sort_values(by=['category', 'filename'], ascending=ascending)
        elif sort_field in ['num_vertices', 'num_faces']:
            df[sort_field] = df[sort_field].fillna(0)
            df = df.sort_values(by=sort_field, ascending=ascending)
        
        df = df.reset_index(drop=True)
        
        # Find the item closest to average
        selected_idx = None
        if button_id == 'avg-vertices-btn' and 'num_vertices' in df.columns:
            valid = df['num_vertices'].dropna()
            if not valid.empty:
                avg_v = valid.mean()
                idx = (df['num_vertices'] - avg_v).abs().idxmin()
                selected_idx = int(idx)
        elif button_id == 'avg-faces-btn' and 'num_faces' in df.columns:
            valid = df['num_faces'].dropna()
            if not valid.empty:
                avg_f = valid.mean()
                idx = (df['num_faces'] - avg_f).abs().idxmin()
                selected_idx = int(idx)
        
        if selected_idx is not None:
            # Get the selected row and create shape info
            row = df.iloc[selected_idx]
            # No toast here - handled by client-side callbacks
            
            try:
                mesh = ShapeMesh.from_file_row(row)
                info = mesh.get_card_header_html()
                # Return in correct order: selected-file-store, shape-info, toast-store
                return {'filename': row['filename'], 'dataset': selected_dataset}, info, dash.no_update
            except Exception as e:
                err_info = html.Div([
                    html.H4("❌ Error Loading Average Shape", style={'color': '#e74c3c', 'marginBottom': '15px'}),
                    html.Div([html.Strong("📄 File: "), row['filepath']], style={'marginBottom': '8px'}),
                    html.Div([html.Strong("⚠️ Error: "), str(e)], style={'color': '#e74c3c'})
                ])
                error_toast_data = dash.no_update  # No old toast system
                return {'filename': row['filename'], 'dataset': selected_dataset}, err_info, error_toast_data
        
        return no_update, no_update, no_update

    # Show loading indicator immediately when average buttons are clicked
    app.clientside_callback(
        """
        function(vertices_clicks, faces_clicks) {
            const ctx = window.dash_clientside.callback_context;
            if (!ctx.triggered.length) {
                return window.dash_clientside.no_update;
            }
            
            // Show loading indicator
            return {'display': 'block'};
        }
        """,
        Output('navigation-loading', 'style'),
        [Input('avg-vertices-btn', 'n_clicks'),
         Input('avg-faces-btn', 'n_clicks')],
        prevent_initial_call=True
    )

    # Hide loading indicator when shape info is updated (navigation complete)
    app.clientside_callback(
        """
        function(shape_info) {
            if (!shape_info || shape_info.length === 0) {
                return window.dash_clientside.no_update;
            }
            
            // Hide loading indicator when navigation completes
            return {'display': 'none'};
        }
        """,
        Output('navigation-loading', 'style', allow_duplicate=True),
        Input('shape-info', 'children'),
        prevent_initial_call=True
    )

    # Client-side callback to scroll to selected file in the list (with lazy loading support)
    app.clientside_callback(
        """
        function(selectedFileData) {
            console.log('Client-side scroll callback triggered with selectedFileData:', selectedFileData);
            
            function findAndScrollToFile() {
                // Find the file list container
                let fileListContainer = document.querySelector('#file-list .file-list-panel');
                if (!fileListContainer) {
                    fileListContainer = document.querySelector('#file-list > div');
                }
                if (!fileListContainer) {
                    fileListContainer = document.querySelector('.file-list-panel');
                }
                
                if (!fileListContainer) {
                    console.log('File list container not found');
                    return false;
                }
                
                // Find all file buttons
                let fileButtons = fileListContainer.querySelectorAll('button[data-filename]');
                console.log('Found', fileButtons.length, 'file buttons');
                
                // Always clear existing selections first
                fileButtons.forEach(btn => {
                    btn.classList.remove('selected-file');
                    btn.classList.remove('file-button-selected');
                    btn.style.backgroundColor = '';
                    btn.style.borderColor = '';
                    btn.style.color = '';
                    btn.style.boxShadow = '';
                    btn.style.border = '';
                });
                
                // If no file is selected, just clear all selections
                if (!selectedFileData || selectedFileData === null || selectedFileData === 'null') {
                    console.log('No file selected, cleared all selections');
                    return true;
                }
                
                // Extract filename from selectedFileData
                let targetFilename = null;
                if (typeof selectedFileData === 'object' && selectedFileData.filename) {
                    targetFilename = selectedFileData.filename;
                }
                
                if (!targetFilename) {
                    console.log('No target filename found');
                    return true;
                }
                
                // Find the button with matching filename
                let targetButton = null;
                fileButtons.forEach(btn => {
                    if (btn.getAttribute('data-filename') === targetFilename) {
                        targetButton = btn;
                    }
                });
                
                if (targetButton) {
                    console.log('Found target button for:', targetFilename);
                    // Apply selected styling
                    targetButton.classList.add('selected-file');
                    targetButton.style.backgroundColor = '#e3f2fd';
                    targetButton.style.borderColor = '#2196f3';
                    targetButton.style.color = '#1976d2';
                    targetButton.style.boxShadow = '0 2px 8px rgba(33, 150, 243, 0.3)';
                    targetButton.style.border = '2px solid #2196f3';
                    
                    // Scroll to the button
                    targetButton.scrollIntoView({
                        behavior: 'smooth',
                        block: 'center'
                    });
                    return true;
                } else {
                    console.log('Target button not found for:', targetFilename, '- may need to load more files');
                    return false;
                }
            }
            
            // First attempt to find and scroll
            setTimeout(function() {
                const found = findAndScrollToFile();
                
                if (!found && selectedFileData && selectedFileData.filename) {
                    // If not found, try loading more files automatically
                    console.log('Attempting to load more files to find target...');
                    const loadMoreBtn = document.getElementById('load-more-btn');
                    if (loadMoreBtn) {
                        const hasMore = loadMoreBtn.style.getPropertyValue('data-has-more') === 'true';
                        if (hasMore) {
                            loadMoreBtn.click();
                            
                            // Try again after loading more files
                            setTimeout(function() {
                                findAndScrollToFile();
                            }, 1000);
                        }
                    }
                }
            }, 100);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('dummy-div', 'children'),
        Input('selected-file-store', 'data'),
        prevent_initial_call=True
    )

    # Client-side callback to clear selection when dataset changes
    app.clientside_callback(
        """
        function(dataset) {
            console.log('Dataset change callback triggered with dataset:', dataset);
            
            // Wait a bit for the DOM to be ready
            setTimeout(function() {
                // Find all file buttons and clear selection styling
                let fileListContainer = document.querySelector('#file-list .file-list-panel');
                if (!fileListContainer) {
                    fileListContainer = document.querySelector('#file-list > div');
                }
                if (!fileListContainer) {
                    fileListContainer = document.querySelector('.file-list-panel');
                }
                
                if (fileListContainer) {
                    let fileButtons = fileListContainer.querySelectorAll('button.file-button');
                    if (fileButtons.length === 0) {
                        fileButtons = fileListContainer.querySelectorAll('button[id*="file-btn"]');
                    }
                    if (fileButtons.length === 0) {
                        fileButtons = fileListContainer.querySelectorAll('button');
                    }
                    
                    console.log('Dataset changed - clearing', fileButtons.length, 'file button selections');
                    
                    // Force clear all selections
                    fileButtons.forEach(btn => {
                        btn.classList.remove('selected-file');
                        btn.classList.remove('file-button-selected'); // Clear both selection classes
                        btn.style.backgroundColor = '';
                        btn.style.borderColor = '';
                        btn.style.color = '';
                        btn.style.boxShadow = '';
                        btn.style.border = '';
                    });
                }
            }, 200); // Slightly longer delay to ensure file list is updated
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('shape-info', 'id'),  # Dummy output
        Input('selected-dataset-store', 'data'),
        prevent_initial_call=True
    )

    # Immediate client-side callback to clear selection when dataset selector changes
    app.clientside_callback(
        """
        function(dataset_value) {
            console.log('Immediate dataset selector change:', dataset_value);
            
            // Immediately clear all selections when dataset dropdown changes
            setTimeout(function() {
                let fileButtons = document.querySelectorAll('button.file-button, button[id*="file-btn"]');
                console.log('Immediate clear - found', fileButtons.length, 'buttons');
                
                fileButtons.forEach(btn => {
                    btn.classList.remove('selected-file');
                    btn.classList.remove('file-button-selected'); // Clear both selection classes
                    btn.style.backgroundColor = '';
                    btn.style.borderColor = '';
                    btn.style.color = '';
                    btn.style.boxShadow = '';
                    btn.style.border = '';
                });
            }, 50); // Very short delay
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('dataset-selector', 'id'),  # Dummy output
        Input('dataset-selector', 'value'),
        prevent_initial_call=True
    )

    # Client-side callback to show and auto-hide toast notifications
    app.clientside_callback(
        """
        function(toasts) {
            if (!toasts || toasts.length === 0) {
                return window.dash_clientside.no_update;
            }
            
            // Show toasts with animation and auto-hide
            setTimeout(function() {
                const toastElements = document.querySelectorAll('.toast:not(.show)');
                toastElements.forEach(function(toast, index) {
                    setTimeout(function() {
                        if (toast && toast.classList) {
                            toast.classList.add('show');
                        }
                    }, index * 100); // Stagger animations
                });
                
                // Auto-hide after 3 seconds - just hide, don't remove (let React manage DOM)
                setTimeout(function() {
                    toastElements.forEach(function(toast) {
                        if (toast && toast.classList) {
                            toast.classList.remove('show');
                            toast.style.opacity = '0';
                            toast.style.transform = 'translateX(-100%)';
                        }
                    });
                }, 3000);
            }, 100);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('toast-container', 'id'),  # Dummy output
        Input('toast-container', 'children'),
        prevent_initial_call=True
    )

    # Dynamic file list height adjustment
    app.clientside_callback(
        """
        function(file_list_content) {
            function adjustFileListHeight() {
                const fileListPanel = document.querySelector('.file-list-panel');
                if (!fileListPanel) return;
                
                const sidePanel = document.querySelector('.side-panel');
                if (!sidePanel) return;
                
                const sidePanelRect = sidePanel.getBoundingClientRect();
                
                // Find the file list panel's position relative to the side panel
                const fileListRect = fileListPanel.getBoundingClientRect();
                const fileListOffsetFromPanelTop = fileListRect.top - sidePanelRect.top;
                
                // Account for side panel padding bottom
                const sidePanelStyle = getComputedStyle(sidePanel);
                const sidePanelPaddingBottom = parseInt(sidePanelStyle.paddingBottom || 0);
                
                // Account for file list panel margins and padding
                const fileListStyle = getComputedStyle(fileListPanel);
                const fileListPaddingBottom = parseInt(fileListStyle.paddingBottom || 0);
                const fileListMarginBottom = parseInt(fileListStyle.marginBottom || 0);
                const fileListBorderBottom = parseInt(fileListStyle.borderBottomWidth || 0);
                
                // IMPORTANT: Reserve space for the load more button container
                const loadMoreContainer = document.querySelector('.load-more-container');
                let loadMoreHeight = 0;
                if (loadMoreContainer) {
                    const loadMoreRect = loadMoreContainer.getBoundingClientRect();
                    loadMoreHeight = loadMoreRect.height;
                    console.log('- Load more button height:', loadMoreHeight);
                }
                // If button not found, estimate it (button + padding + margins)
                if (loadMoreHeight === 0) {
                    loadMoreHeight = 50; // Estimated: button (28px) + margins (10px top + 10px bottom) + padding
                    console.log('- Load more button not found, using estimated height:', loadMoreHeight);
                }
                
                // Calculate available height: total panel height minus offset from top minus bottom spacing minus button space
                const bottomSpacing = sidePanelPaddingBottom + fileListPaddingBottom + fileListMarginBottom + fileListBorderBottom + loadMoreHeight;
                const availableHeight = sidePanelRect.height - fileListOffsetFromPanelTop - bottomSpacing;
                
                // Add some buffer to prevent overflow
                const buffer = 20; // Increased buffer for safety
                const minHeight = 200;
                const maxHeight = Math.max(minHeight, availableHeight - buffer);
                
                // Apply the calculated height
                fileListPanel.style.height = maxHeight + 'px';
                fileListPanel.style.maxHeight = maxHeight + 'px';
                
                console.log('File list height calculation:');
                console.log('- Side panel height:', sidePanelRect.height);
                console.log('- File list offset from panel top:', fileListOffsetFromPanelTop);
                console.log('- Bottom spacing needed (including button):', bottomSpacing);
                console.log('- Available height:', availableHeight);
                console.log('- Buffer applied:', buffer);
                console.log('- Final height applied:', maxHeight + 'px');
            }
            
            // Adjust on content change with a delay to ensure DOM is ready
            setTimeout(adjustFileListHeight, 150);
            
            // Also adjust on window resize
            if (!window.fileListResizeListener) {
                window.fileListResizeListener = true;
                window.addEventListener('resize', function() {
                    setTimeout(adjustFileListHeight, 150);
                });
            }
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('file-list', 'className'),  # Dummy output
        Input('file-list', 'children'),
        prevent_initial_call=True
    )

    # Auto-scroll file list to top when dataset or filters change
    app.clientside_callback(
        """
        function(dataset, category, filename, vertices_op, vertices_val, faces_op, faces_val, sort_field, sort_order) {
            // Wait for DOM to be ready
            setTimeout(function() {
                const fileListPanel = document.querySelector('.file-list-panel');
                if (fileListPanel) {
                    console.log('Scrolling file list to top due to filter/dataset change');
                    fileListPanel.scrollTop = 0; // Scroll to top
                }
            }, 100);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('file-list', 'id'),  # Dummy output
        [Input('selected-dataset-store', 'data'),
         Input('category-filter', 'value'),
         Input('filename-filter', 'value'),
         Input('vertices-operator', 'value'),
         Input('vertices-value', 'value'),
         Input('faces-operator', 'value'),
         Input('faces-value', 'value'),
         Input('sort-field', 'value'),
         Input('sort-order', 'data-order')],
        prevent_initial_call=True
    )

    # Toast system using stores (no DOM conflicts)
    @app.callback(
        [Output('toast-container', 'children'),
         Output('toast-container', 'className'),
         Output('toast-interval', 'disabled'),
         Output('toast-interval', 'n_intervals')],
        Input('toast-store', 'data'),
        prevent_initial_call=True
    )
    def show_toast(toast_data):
        """Show toast notification from store data"""
        print(f"🔔 Toast callback triggered with data: {toast_data}")  # Debug
        
        if not toast_data or not toast_data.get('message'):
            print("❌ No toast data or message")  # Debug
            return [], "toast-container", True, 0
        
        print(f"✅ Creating toast: {toast_data['message']}")  # Debug
        toast_element = html.Div([
            html.Span(toast_data['icon'], className="toast-icon"),
            html.Span(toast_data['message'], className="toast-message")
        ], className=f"toast {toast_data['type']}")
        
        # Check if this is a missing step toast (ONLY warning toasts with specific missing step message)
        message = toast_data.get('message', '').lower()
        is_missing_step = (toast_data.get('icon') == '⚠️' and 
                          toast_data.get('type') == 'warning' and
                          'missing' in message and 
                          ('step' in message or 'showing' in message))
        container_class = "toast-container missing-step-position" if is_missing_step else "toast-container"
        
        print(f"🔍 Toast detection - Icon: '{toast_data.get('icon')}', Type: '{toast_data.get('type')}', Message: '{toast_data.get('message')}'")
        print(f"📍 Is missing step: {is_missing_step}, Container class: {container_class}")  # Debug
        
        return [toast_element], container_class, False, 0  # Enable interval and reset counter

    @app.callback(
        [Output('toast-container', 'children', allow_duplicate=True),
         Output('toast-interval', 'disabled', allow_duplicate=True)],
        Input('toast-interval', 'n_intervals'),
        State('toast-interval', 'disabled'),
        prevent_initial_call=True
    )
    def clear_toast_after_delay(n_intervals, interval_disabled):
        """Clear toast after 80 intervals (4 seconds at 50ms)"""
        if interval_disabled:
            return no_update, no_update
        
        if n_intervals >= 80:  # 4 seconds at 50ms intervals
            return [], True  # Clear toast and disable interval
        
        return no_update, no_update

    # Step Toast System - positioned over 3D viewer
    @app.callback(
        [Output('step-toast-container', 'children'),
         Output('step-toast-interval', 'disabled'),
         Output('step-toast-interval', 'n_intervals')],
        Input('step-toast-store', 'data'),
        prevent_initial_call=True
    )
    def show_step_toast(step_toast_data):
        """Show step toast notification positioned over 3D viewer"""
        print(f"🔔 Step toast callback triggered with data: {step_toast_data}")  # Debug
        
        if not step_toast_data or not step_toast_data.get('message'):
            print("❌ No step toast data or message")  # Debug
            return [], True, 0
        
        print(f"✅ Creating step toast: {step_toast_data['message']}")  # Debug
        toast_element = html.Div([
            html.Span(step_toast_data['icon'], className="toast-icon"),
            html.Span(step_toast_data['message'], className="toast-message")
        ], className=f"toast {step_toast_data['type']} show")
        
        return [toast_element], False, 0  # Enable interval and reset counter

    @app.callback(
        [Output('step-toast-container', 'children', allow_duplicate=True),
         Output('step-toast-interval', 'disabled', allow_duplicate=True)],
        Input('step-toast-interval', 'n_intervals'),
        State('step-toast-interval', 'disabled'),
        prevent_initial_call=True
    )
    def clear_step_toast_after_delay(n_intervals, interval_disabled):
        """Clear step toast after 40 intervals (4 seconds at 100ms)"""
        if interval_disabled:
            return no_update, no_update
        
        if n_intervals >= 40:  # 4 seconds (shorter duration for step messages)
            return [], True  # Clear toast and disable interval
        
        return no_update, no_update

    # 1) File list render and data preparation
    @app.callback(
        [Output('file-list', 'children'),
         Output('file-data-store', 'data'),
         Output('current-batch-store', 'data'),
         Output('load-more-btn', 'style'),
         Output('file-count-info', 'children'),
         Output('load-more-btn', 'data-has-more')],
        [Input('category-filter', 'value'),
         Input('filename-filter', 'value'),
         Input('vertices-operator', 'value'),
         Input('vertices-value', 'value'),
         Input('faces-operator', 'value'),
         Input('faces-value', 'value'),
         Input('sort-field', 'value'),
         Input('sort-order', 'data-order'),
         Input('selected-dataset-store', 'data')],
        prevent_initial_call=True
    )
    def update_file_list(selected_category, filename_filter, vertices_op, vertices_val, faces_op, faces_val, sort_field, sort_order, selected_dataset):        
        """
        Render the list of files based on current filters and sorting.
        Optimized to avoid slow analysis computation during dataset switching.
        """
        return update_file_list_internal('none', selected_category, filename_filter, vertices_op, vertices_val, faces_op, faces_val, sort_field, sort_order, selected_dataset)

    def create_file_button(item):
        """Create a file button from item data"""
        import base64
        import pandas as pd
        
        raw_vertices = item.get('num_vertices', 0)
        raw_faces = item.get('num_faces', 0)
        vertices_count = f"{int(raw_vertices):,}" if pd.notna(raw_vertices) and raw_vertices > 0 else "N/A"
        faces_count = f"{int(raw_faces):,}" if pd.notna(raw_faces) and raw_faces > 0 else "N/A"
        
        # Encode filename as base64 to avoid JSON parsing issues
        encoded_filename = base64.b64encode(item['filename'].encode('utf-8')).decode('ascii')
        
        return html.Button(
            html.Div([
                html.Div([
                    html.Strong(f"📁 {item['category']}", className="category-text", style={'fontSize': '0.9em'}),
                    html.Span(f" | 📄 {item['filename']}", className="filename-text", style={'fontSize': '0.85em', 'color': '#555'})
                ], style={'marginBottom': '2px'}),
                html.Div([
                    html.Span(f"🔺 Vertices: {vertices_count}", className="stats-text", 
                            style={'marginRight': '8px', 'fontSize': '0.75em', 'color': '#888'}),
                    html.Span(f"🔷 Faces: {faces_count}", className="stats-text", 
                            style={'fontSize': '0.75em', 'color': '#888'})
                ])
            ]),
            id={'type': 'file-btn', 'filename': encoded_filename},
            className='file-button',
            n_clicks=0,
            **{'data-filename': item['filename'], 'data-file-index': item['original_index'], 'data-category': item['category']}
        )

    def update_file_list_internal(avg_filter, selected_category, filename_filter, vertices_op, vertices_val, faces_op, faces_val, sort_field, sort_order, selected_dataset):        
        """
        Render the list of files based on current filters and sorting with lazy loading support.
        Returns: [file_items, full_data, current_batch, load_more_style, file_count_info, has_more_attr]
        """
        if selected_dataset is None or selected_dataset == "":
            selected_dataset = 'Data'

        # Use high-performance cached dataset (already includes merged analysis data)
        file_df = get_cached_dataset_data(selected_dataset)

        if file_df.empty:
            return [html.P("❌ No files found in Data directory",
                           style={'color': 'red', 'textAlign': 'center'})], [], 0, {'display': 'none'}, "📊 No files", 'false'

        # Cached data already includes analysis columns (num_vertices, num_faces)
        print(f"✅ Using cached data for {selected_dataset} with {len(file_df)} shapes")
        
        # Apply filters exactly like before
        df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
        
        # Apply filename filtering if provided
        if filename_filter and filename_filter.strip() and not df.empty and 'filename' in df.columns:
            try:
                import fnmatch
                pattern = filename_filter.strip()
                df = df[df['filename'].apply(lambda x: fnmatch.fnmatch(x.lower(), pattern.lower()))]
            except Exception as e:
                print(f"❌ Error applying filename filter '{filename_filter}': {e}")

        # Apply vertices filtering if provided
        if vertices_val is not None and vertices_val != '' and not df.empty and 'num_vertices' in df.columns:
            try:
                vertices_val = int(vertices_val)
                if vertices_op == 'eq':
                    df = df[df['num_vertices'] == vertices_val]
                elif vertices_op == 'gt':
                    df = df[df['num_vertices'] > vertices_val]
                elif vertices_op == 'lt':
                    df = df[df['num_vertices'] < vertices_val]
            except (ValueError, TypeError) as e:
                print(f"❌ Error applying vertices filter '{vertices_val}': {e}")

        # Apply faces filtering if provided
        if faces_val is not None and faces_val != '' and not df.empty and 'num_faces' in df.columns:
            try:
                faces_val = int(faces_val)
                if faces_op == 'eq':
                    df = df[df['num_faces'] == faces_val]
                elif faces_op == 'gt':
                    df = df[df['num_faces'] > faces_val]
                elif faces_op == 'lt':
                    df = df[df['num_faces'] < faces_val]
            except (ValueError, TypeError) as e:
                print(f"❌ Error applying faces filter '{faces_val}': {e}")

        # Apply sorting
        ascending = True if sort_order == 'asc' else False
        df = df.copy()
        if sort_field == 'category':
            df = df.sort_values(by=['category', 'filename'], ascending=ascending)
        elif sort_field in ['num_vertices', 'num_faces']:
            if sort_field not in df.columns:
                return [html.Div([
                    html.P("⚠️ Sorting Not Available", style={
                        'color': '#f39c12', 'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '10px'
                    }),
                    html.P(f"Cannot sort by {sort_field} - analysis data not available for this dataset.", style={
                        'color': '#7f8c8d', 'textAlign': 'center', 'marginBottom': '5px'
                    })
                ])], [], 0, {'display': 'none'}, "📊 Sorting unavailable", 'false'
            df[sort_field] = df[sort_field].fillna(0)
            df = df.sort_values(by=sort_field, ascending=ascending)

        # Apply average filtering
        if avg_filter == 'avg_faces' and 'num_faces' in df.columns and not df.empty:
            valid = df['num_faces'].dropna()
            if not valid.empty:
                avg_f = valid.mean()
                idx = (df['num_faces'] - avg_f).abs().idxmin()
                if idx in df.index:
                    df = df.loc[[idx]].reset_index(drop=True)
        elif avg_filter == 'avg_vertices' and 'num_vertices' in df.columns and not df.empty:
            valid = df['num_vertices'].dropna()
            if not valid.empty:
                avg_v = valid.mean()
                idx = (df['num_vertices'] - avg_v).abs().idxmin()
                if idx in df.index:
                    df = df.loc[[idx]].reset_index(drop=True)

        # Keep track of original indices for proper file selection mapping
        original_indices = df.index.tolist()
        df = df.reset_index(drop=True)

        # LAZY LOADING: Prepare complete dataset for store
        full_data = []
        for idx, (df_idx, row) in enumerate(df.iterrows()):
            full_data.append({
                'idx': idx,
                'original_index': original_indices[idx],
                'filename': row['filename'],
                'category': row['category'],
                'num_vertices': row.get('num_vertices', 0),
                'num_faces': row.get('num_faces', 0)
            })

        # Initial batch - first 150 files
        batch_size = 150
        initial_batch = full_data[:batch_size]
        
        # Create file count info for top-left corner
        total_files = len(full_data)
        if total_files > batch_size:
            file_count_info = f"📊 {batch_size} of {total_files:,} files"
        else:
            file_count_info = f"📊 {total_files:,} files"
        
        # Create file items without header (header is now in top-left corner)
        file_items = []
        for item in initial_batch:
            file_items.append(create_file_button(item))
        
        # Show load-more button if there are more files (back to manual loading)
        # No longer using infinite scroll
        has_more_files = len(full_data) > batch_size
        load_more_style = {'display': 'block', 'margin': '10px auto'} if has_more_files else {'display': 'none'}
        has_more_attr = 'true' if has_more_files else 'false'
        
        return file_items, full_data, batch_size, load_more_style, file_count_info, has_more_attr

    # Load more files callback
    @app.callback(
        [Output('file-list', 'children', allow_duplicate=True),
         Output('current-batch-store', 'data', allow_duplicate=True),
         Output('load-more-btn', 'style', allow_duplicate=True),
         Output('file-count-info', 'children', allow_duplicate=True),
         Output('load-more-btn', 'data-has-more', allow_duplicate=True)],
        [Input('load-more-btn', 'n_clicks')],
        [State('file-data-store', 'data'),
         State('current-batch-store', 'data')],
        prevent_initial_call=True
    )
    def load_more_files(n_clicks, full_data, current_batch):
        """Load the next batch of files"""
        if not n_clicks or not full_data:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        batch_size = 150
        new_batch = current_batch + batch_size
        total_files = len(full_data)
        
        # Get all files up to the new batch
        files_to_show = full_data[:new_batch]
        
        # Create file items without header (header is now in top-left corner)
        file_items = []
        for item in files_to_show:
            file_items.append(create_file_button(item))
        
        # Update file count info for top-left corner
        remaining_files = total_files - new_batch
        if remaining_files > 0:
            file_count_info = f"📊 {new_batch} of {total_files:,} files"
        else:
            file_count_info = f"📊 All {total_files:,} files"
        
        # Show load-more button if there are more files (back to manual loading)
        # No longer using infinite scroll  
        has_more_files = remaining_files > 0
        load_more_style = {'display': 'block', 'margin': '10px auto'} if has_more_files else {'display': 'none'}
        has_more_attr = 'true' if has_more_files else 'false'
        
        return file_items, new_batch, load_more_style, file_count_info, has_more_attr

    # Client-side infinite scroll detection - TEMPORARILY DISABLED
    # app.clientside_callback(
    #     """
    #     function(n_intervals) {
    #         return window.dash_clientside.no_update;
    #     }
    #     """,
    #     Output('scroll-sentinel', 'id'),
    #     Input('scroll-interval', 'n_intervals'),
    #     prevent_initial_call=True
    # )

    # 2) Selected file highlight (client-side)
    app.clientside_callback(
        """
        function(selectedFileIdx) {
            console.log('Selection callback triggered with index:', selectedFileIdx);
            
            // Always clear all selections first
            const allButtons = document.querySelectorAll('[data-file-index]');
            console.log('Found file buttons:', allButtons.length);
            
            allButtons.forEach(button => {
                button.classList.remove('file-button-selected');
                button.classList.remove('selected-file'); // Clear both selection classes
            });
            
            // If no file is selected (null/undefined), just return after clearing
            if (selectedFileIdx == null || selectedFileIdx === undefined || selectedFileIdx === 'null') {
                console.log('No file selected, cleared all selections');
                return window.dash_clientside.no_update;
            }
            
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

    # 3) Click handler -> loads file, updates info + selected filename
    @app.callback(
        [Output('shape-info', 'children'),
         Output('selected-file-store', 'data')],
        [Input({'type': 'file-btn', 'filename': dash.dependencies.ALL}, 'n_clicks'),
         Input('category-filter', 'value'),
         Input('filename-filter', 'value'),
         Input('vertices-operator', 'value'),
         Input('vertices-value', 'value'),
         Input('faces-operator', 'value'),
         Input('faces-value', 'value'),
         Input('sort-field', 'value'),
         Input('sort-order', 'data-order'),
         Input('selected-dataset-store', 'data')],
        prevent_initial_call=True
    )
    def select_or_reset_file(n_clicks_list, selected_category, filename_filter, vertices_op, vertices_val, faces_op, faces_val, sort_field, sort_order, selected_dataset):
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
                prop_id_str = prop_id.split('.')[0]
                print(f"[DEBUG] Raw prop_id: {prop_id_str}")
                comp_id = json.loads(prop_id_str)
                
                # Decode the base64 encoded filename
                import base64
                encoded_filename = comp_id['filename']
                clicked_filename = base64.b64decode(encoded_filename).decode('utf-8')
                print(f"[DEBUG] Decoded filename: {clicked_filename}")
                
                # Get the full dataset to find the index of this filename
                file_df = get_cached_dataset_data(selected_dataset)
                if file_df is None or file_df.empty:
                    return html.P("❌ No data available"), None
                
                # Find the index of the clicked filename in the original dataset
                matching_rows = file_df[file_df['filename'] == clicked_filename]
                if matching_rows.empty:
                    return html.P(f"❌ File {clicked_filename} not found"), None
                
                file_idx = matching_rows.index[0]  # Get the original DataFrame index
                print(f"🎯 Clicked filename: {clicked_filename}, found at index: {file_idx}")
                
            except Exception as e:
                print(f"❌ Error parsing clicked file: {e}")
                print(f"[DEBUG] Failed prop_id: {prop_id}")
                return html.P(f"❌ Error processing file click: {e}"), None

            # Rebuild file_df for current filters
            if selected_dataset is None or selected_dataset == "":
                selected_dataset = 'Data'
            # Use high-performance cached dataset (already merged)
            file_df_local = get_cached_dataset_data(selected_dataset)
            df = file_df_local if selected_category == 'all' else file_df_local[file_df_local['category'] == selected_category]
            
            # Apply filename filtering if provided
            if filename_filter and filename_filter.strip() and not df.empty and 'filename' in df.columns:
                try:
                    import fnmatch
                    pattern = filename_filter.strip()
                    df = df[df['filename'].apply(lambda x: fnmatch.fnmatch(x.lower(), pattern.lower()))]
                except Exception as e:
                    print(f"❌ Error applying filename filter '{filename_filter}': {e}")
                    # Continue without filename filtering if there's an error

            # Apply vertices filtering if provided
            if vertices_val is not None and vertices_val != '' and not df.empty and 'num_vertices' in df.columns:
                try:
                    vertices_val = int(vertices_val)
                    if vertices_op == 'eq':
                        df = df[df['num_vertices'] == vertices_val]
                    elif vertices_op == 'gt':
                        df = df[df['num_vertices'] > vertices_val]
                    elif vertices_op == 'lt':
                        df = df[df['num_vertices'] < vertices_val]
                except (ValueError, TypeError) as e:
                    print(f"❌ Error applying vertices filter '{vertices_val}': {e}")
                    # Continue without vertices filtering if there's an error

            # Apply faces filtering if provided
            if faces_val is not None and faces_val != '' and not df.empty and 'num_faces' in df.columns:
                try:
                    faces_val = int(faces_val)
                    if faces_op == 'eq':
                        df = df[df['num_faces'] == faces_val]
                    elif faces_op == 'gt':
                        df = df[df['num_faces'] > faces_val]
                    elif faces_op == 'lt':
                        df = df[df['num_faces'] < faces_val]
                except (ValueError, TypeError) as e:
                    print(f"❌ Error applying faces filter '{faces_val}': {e}")
                    # Continue without faces filtering if there's an error
            
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
            
            df = df.reset_index(drop=True)
            
            # Find the clicked file in the filtered dataset using filename
            matching_filtered_rows = df[df['filename'] == clicked_filename]
            if matching_filtered_rows.empty:
                return html.P(f"❌ File {clicked_filename} not found in filtered results"), None
            
            filtered_file_idx = matching_filtered_rows.index[0]  # Get index in filtered dataset
            print(f"🎯 File {clicked_filename} found at filtered index: {filtered_file_idx}")
            
            row = df.iloc[filtered_file_idx]
            try:
                mesh = ShapeMesh.from_file_row(row)
                info = mesh.get_card_header_html()
                # Return the filename instead of index for more robust identification
                return info, {'filename': clicked_filename, 'dataset': selected_dataset}
            except Exception as e:
                err = html.Div([
                    html.H4("❌ Error Loading File", style={'color': '#e74c3c', 'marginBottom': '15px'}),
                    html.Div([html.Strong("📄 File: "), row['filepath']], style={'marginBottom': '8px'}),
                    html.Div([html.Strong("⚠️ Error: "), str(e)], style={'color': '#e74c3c'})
                ])
                return err, {'filename': clicked_filename, 'dataset': selected_dataset}
        else:
            # If triggered by filter/sort/dataset change, clear selection
            empty_info = html.P("🔍 Select a 3D shape from the list to view details", className="shape-info-hint")
            return empty_info, None

    # 4) 3D viewer update
    @app.callback(
        [Output('3d-plot', 'figure'),
         Output('toast-store', 'data', allow_duplicate=True),
         Output('step-toast-store', 'data', allow_duplicate=True)],
        [Input('display-options', 'value'),
         Input('selected-file-store', 'data'),
         Input('color-selector', 'value'),
         Input('normalization-toggle', 'value'),
         Input('processing-step-slider', 'value'),
         Input('selected-dataset-store', 'data')],
        [State('category-filter', 'value'),
         State('sort-field', 'value'),
         State('sort-order', 'data-order'),
         Input('3d-plot', 'figure')],
        prevent_initial_call=True
    )
    def update_plot(display_options, 
                    selected_file_data, 
                    mesh_color,                      
                    show_normalized,
                    processing_step,
                    selected_dataset,
                    selected_category, 
                    sort_field, 
                    sort_order,
                    current_fig):
        
        """
        Update the 3D plot based on user selections and current figure state.

        Parameters:
        - display_options: list of str, display options selected (e.g., 'wireframe', 'smooth_shading')
        - selected_file_data: dict with 'filename' and 'dataset', or None if no file selected
        - mesh_color: str, color selected for the mesh
        - show_normalized: list of str, normalization toggle state
        - processing_step: int, processing step index (0-5) for step-by-step viewing
        - selected_dataset: str, currently selected dataset from dropdown
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
        
        if selected_file_data is None:
            return create_3d_plot(np.array([]), np.array([]), "Select a shape to view",
                                  mesh_color=mesh_color or 'lightblue'), no_update, no_update

        # Safety check: ensure selected_file_data is a dictionary
        if not isinstance(selected_file_data, dict):
            print(f"❌ ERROR: selected_file_data should be dict, got {type(selected_file_data)}: {selected_file_data}")
            return create_3d_plot(np.array([]), np.array([]), "Invalid selection data",
                                  mesh_color=mesh_color or 'lightblue'), no_update, no_update

        # Get filename and dataset from the selection data
        selected_filename = selected_file_data.get('filename')
        file_dataset = selected_file_data.get('dataset', selected_dataset)
        
        if not selected_filename:
            return create_3d_plot(np.array([]), np.array([]), "No valid shape selected",
                                  mesh_color=mesh_color or 'lightblue'), no_update, no_update

        if file_dataset is None or file_dataset == "":
            file_dataset = 'Data'
        
        # Use high-performance cached dataset
        file_df = get_cached_dataset_data(file_dataset)

        if file_df is None or file_df.empty:
            return create_3d_plot(np.array([]), np.array([]), "No valid shape selected",
                                  mesh_color=mesh_color or 'lightblue'), None

        # Find the selected file by filename in the dataset
        matching_rows = file_df[file_df['filename'] == selected_filename]
        if matching_rows.empty:
            return create_3d_plot(np.array([]), np.array([]), f"File {selected_filename} not found",
                                  mesh_color=mesh_color or 'lightblue'), None
        
        row = matching_rows.iloc[0]  # Get the first (and should be only) matching row
        print(f"🎯 3D Plot: Loading {selected_filename} from {file_dataset}")
        
        # Determine which file to load based on processing step slider
        step_row = row
        title_suffix = ""
        
        # Only use step processing if we're in a dataset that supports it AND the slider is enabled
        step_fallback_info = None
        if (processing_step is not None and 
            selected_dataset and 
            ('UnifiedPreprocessed' in selected_dataset or 'Normalized' in selected_dataset) and
            row.get('has_processing_steps', False)):
            
            # Special handling for D00355 - force missing step 1 behavior
            if 'D00355' in row.get('filename', '') and processing_step == 1:
                print(f"[DEBUG] D00355 step 1 requested - forcing fallback to step 0 (original)")
                # Force fallback to step 0 (original) since step 1 is missing
                actual_file_path = row['filepath']
                step_fallback_info = {
                    'requested_step': 1,
                    'actual_step': 0,
                    'requested_step_name': 'Remeshed',
                    'actual_step_name': 'Original',
                    'step_available': False
                }
                title_suffix = f" (Original Step - Fallback)"
                
                # Create a temporary row with the step file path
                step_row = row.copy()
                step_row['filepath'] = actual_file_path
                print(f"[DEBUG] D00355: Forced fallback from step 1 to step 0: {actual_file_path}")
            else:
                # Normal step processing
                actual_file_path, actual_step_index, step_info = get_step_file_path(row, processing_step)
                title_suffix = f" ({step_info['name']} Step)"
                
                # Check if we had to use a fallback
                if step_info.get('fallback_used', False):
                    step_fallback_info = {
                        'requested_step': step_info['requested_step'],
                        'actual_step': step_info['actual_step'],
                        'requested_step_name': step_info.get('requested_step_name', 'Unknown'),
                        'actual_step_name': step_info['name'],
                        'step_available': step_info.get('step_available', False)
                    }
                    title_suffix = f" ({step_info['name']} Step - Fallback)"
                
                # Create a temporary row with the step file path
                step_row = row.copy()
                step_row['filepath'] = actual_file_path
                
                print(f"[DEBUG] Loading step {processing_step} -> {actual_step_index} file: {actual_file_path}")
                if step_fallback_info:
                    print(f"[DEBUG] Step fallback: requested {step_fallback_info['requested_step_name']} -> showing {step_fallback_info['actual_step_name']}")
        
        # Create ShapeMesh instance and handle special cases for different datasets
        try:
            # Special handling for NormalizedShapes dataset
            if selected_dataset == 'NormalizedShapes':
                # NormalizedShapes dataset contains pre-normalized files
                mesh = ShapeMesh.from_file_row(step_row)
                vertices = mesh.vertices  # Already normalized
                title_suffix += " (Pre-normalized Dataset)"
                camera_config = None  # Use default camera for normalized shapes
                print(f"[DEBUG] Using pre-normalized vertices from NormalizedShapes dataset for {step_row['filename']}")
            
            # Handle normalization toggle for other datasets (only if not using processing steps)
            elif show_normalized and 'normalized' in show_normalized and not row.get('has_processing_steps', False):
                from core.normalized_cache import normalized_cache
                # Try to load from cache first
                if normalized_cache.is_normalized_available(row['filename'], selected_dataset):
                    mesh = normalized_cache.load_normalized_shape(row['filename'], selected_dataset)
                    vertices = mesh.vertices
                    title_suffix += " (Cached Normalized)"
                    print(f"[DEBUG] Using cached normalized vertices for {row['filename']}")
                else:
                    # Fall back to computing normalization
                    mesh = ShapeMesh.from_file_row(step_row)
                    vertices = mesh.apply_full_normalization()
                    title_suffix += " (Computed Normalized)"
                    print(f"[DEBUG] Computing normalized vertices for {row['filename']} (cache not available)")
                
                camera_config = None  # Use default camera for normalized shapes
            else:
                # Load the selected step file or original shape
                mesh = ShapeMesh.from_file_row(step_row)
                vertices = mesh.vertices
                camera_config = mesh.get_optimal_camera_position()
                print(f"[DEBUG] Using step file for {step_row['filename']}")
            
            faces = mesh.faces
                
        except Exception as e:
            print(f"[DEBUG] ShapeMesh failed: {e}")
            # Fallback to original method if ShapeMesh fails
            file_path_to_use = step_row['filepath'] if 'step_row' in locals() else row['filepath']
            vertices, faces = OBJParser.parse_obj_file(file_path_to_use)
            camera_config = None
            title_suffix += " (Fallback Parser)"
        
        show_wire = 'wireframe' in (display_options or [])
        title = f"{row['category']} - {row['filename']}{title_suffix}"

        fig = create_3d_plot(vertices, faces, title, show_wireframe=show_wire,
                              mesh_color=mesh_color or 'lightblue',
                              smooth_shading=smooth_shading,
                              camera_config=camera_config)
        
        # If user had a previous camera position, restore it
        if camera:
            fig.update_layout(scene_camera=camera)
        
        # Create toast notification for step fallback if needed
        # Handle missing step notification
        regular_toast_data = no_update  # Don't interfere with other toasts
        step_toast_data = no_update     # Don't send empty step toasts
        
        if step_fallback_info:
            if not step_fallback_info['step_available']:
                # Send step missing messages to the step-toast-store (positioned over 3D viewer)
                step_toast_data = create_toast_data(
                    f"ℹ️ Step '{step_fallback_info['requested_step_name']}' is not available for this shape. "
                    f"Displaying '{step_fallback_info['actual_step_name']}' step instead.",
                    "info",
                    "ℹ️"
                )
                print(f"[DEBUG] Created step toast notification for missing step")
            
        return fig, regular_toast_data, step_toast_data


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
                                smooth_shading=smooth_shading,
                                camera_config=None,
                                use_rotated_vertices=False)

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
    
     # Store current dataset in dcc.Store and clear selection when dataset changes
    @app.callback(
        [Output('selected-dataset-store', 'data'),
         Output('selected-file-store', 'data', allow_duplicate=True),
         Output('shape-info', 'children', allow_duplicate=True)],
        Input('dataset-selector', 'value'),
        State('selected-dataset-store', 'data'),
        prevent_initial_call=True
    )
    def update_selected_dataset(selected_dataset, current_dataset):
        """
        Update the selected dataset store when the dropdown changes and clear file selection.

        Parameters:
        - selected_dataset: str, newly selected dataset from dropdown
        - current_dataset: str, currently stored dataset

        Returns:
        - tuple: (str, None, str) - updated dataset value, cleared file selection, and cleared shape info
        """
        if selected_dataset and selected_dataset != current_dataset:
            # Clear both file selection and shape info when dataset changes
            empty_info = html.Div([
                html.P("ℹ️ Select a 3D shape from the list to view details", 
                       style={'color': '#666', 'fontStyle': 'italic', 'textAlign': 'center', 'padding': '20px'})
            ])
            return selected_dataset, None, empty_info  # Clear selected file and shape info when dataset changes
        return current_dataset, no_update, no_update

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
            # Use high-performance cached dataset
            file_df = get_cached_dataset_data(selected_dataset)
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

    # Update normalization toggle options and value when file is selected
    @app.callback(
        [Output('normalization-toggle', 'options'),
         Output('normalization-toggle', 'value')],
        [Input('selected-file-store', 'data'),
         Input('selected-dataset-store', 'data')],
        [State('category-filter', 'value'),
         State('sort-field', 'value'),
         State('sort-order', 'data-order')],
        prevent_initial_call=True
    )
    def update_normalization_toggle(selected_file_data, selected_dataset, selected_category, sort_field, sort_order):
        """
        Update the normalization toggle options and value based on the selected file's filename.
        Automatically check normalization if filename contains '_normalized' suffix.

        Parameters:
        - selected_file_idx: int or None, index of the selected file from the file list
        - selected_dataset: str, currently selected dataset from dropdown
        - avg_filter: str, average filter option ('none', 'avg_faces', 'avg_vertices')
        - selected_category: str, selected category filter ('all' or specific category)
        - sort_field: str, field to sort by ('category', 'num_vertices', 'num_faces')
        - sort_order: str, sort order ('asc' or 'desc')

        Returns:
        - options: list of dict, options for the normalization toggle
        - value: list, updated value for the normalization toggle checklist
        """
        # Define the standard options for normalization toggle (always disabled)
        options = [{'label': '', 'value': 'normalized', 'disabled': True}]
        
        # If no file is selected, uncheck the toggle
        if selected_file_data is None:
            return options, []
        
        # Extract filename from the selection data
        if isinstance(selected_file_data, dict):
            selected_filename = selected_file_data.get('filename')
        else:
            # Fallback for old format (shouldn't happen, but just in case)
            return options, []
        
        if not selected_filename:
            return options, []
        
        try:
            # Check if filename contains '_normalized' (case-insensitive)
            if '_normalized' in selected_filename.lower() or '_unified' in selected_filename.lower():
                # For normalized files, disable the checkbox and check it
                return options, ['normalized']  # Check the normalization toggle
            else:
                # For non-normalized files, disable the checkbox and uncheck it
                return options, []  # Uncheck the normalization toggle
                
        except Exception as e:
            print(f"[DEBUG] Error updating normalization toggle: {e}")
            return options, []

    # Step slider control callbacks
    @app.callback(
        Output('display-step-panel', 'style'),
        Input('selected-dataset-store', 'data'),
        prevent_initial_call=True
    )
    def update_step_panel_visibility(selected_dataset):
        """
        Show/hide the step panel based on dataset type.
        Only show for datasets that contain processed step files.
        """
        if selected_dataset and ('UnifiedPreprocessed' in selected_dataset or 'Normalized' in selected_dataset):
            return {'display': 'block'}
        else:
            return {'display': 'none'}

    @app.callback(
        [Output('processing-step-slider', 'disabled'),
         Output('processing-step-slider', 'value'),
         Output('processing-step-slider', 'max'),
         Output('step-info-display', 'children'),
         Output('step-info-display', 'className')],
        [Input('selected-file-store', 'data'),
         Input('selected-dataset-store', 'data')],
        prevent_initial_call=True
    )
    def update_step_slider_state(selected_file_data, selected_dataset):
        """
        Enable/disable the step slider based on whether the selected shape has processing steps.
        Updates the slider max value based on available steps for the selected shape.
        """
        # First check if we're in a dataset that should show step controls
        if not selected_dataset or not ('UnifiedPreprocessed' in selected_dataset or 'Normalized' in selected_dataset):
            return True, 5, 5, "Not available for this dataset", "step-info-text"
            
        if selected_file_data is None:
            return True, 5, 5, "Select a processed shape to enable step navigation", "step-info-text"
        
        # Extract filename from the selection data
        if isinstance(selected_file_data, dict):
            selected_filename = selected_file_data.get('filename')
            file_dataset = selected_file_data.get('dataset', selected_dataset)
        else:
            return True, 5, 5, "Invalid shape selection", "step-info-text"
        
        if not selected_filename:
            return True, 5, 5, "Invalid shape selection", "step-info-text"
        
        try:
            # Get the file data for the selected shape
            file_df = get_cached_dataset_data(file_dataset)
            if file_df is None or file_df.empty:
                return True, 5, 5, "Invalid shape selection", "step-info-text"
            
            # Find the file by filename
            matching_rows = file_df[file_df['filename'] == selected_filename]
            if matching_rows.empty:
                return True, 5, 5, "Shape not found in dataset", "step-info-text"
            
            row = matching_rows.iloc[0]
            
            # Check if shape has processing steps
            if not row.get('has_processing_steps', False):
                return True, 5, 5, "This shape has no processing steps available", "step-info-text"
            
            # Get available step information
            step_availability = get_available_steps(row)
            available_indices = step_availability['available_step_indices']
            recommended_max = step_availability['recommended_max_step']
            
            if not available_indices:
                return True, 5, 5, "No processing steps found for this shape", "step-info-text"
            
            # Set slider max to the highest available step
            slider_max = max(available_indices)
            
            # Set initial value to the recommended max step (usually the final step)
            initial_value = recommended_max
            
            # Create info message showing available steps
            missing_steps = step_availability['missing_step_indices']
            if missing_steps:
                step_names = ["Orig", "Mesh", "Trans", "Align", "Flip", "Scale"]
                available_names = [step_names[i] for i in available_indices]
                missing_names = [step_names[i] for i in missing_steps if i < len(step_names)]
                info_msg = f"Available steps: {', '.join(available_names)}"
                if missing_names:
                    info_msg += f" (Missing: {', '.join(missing_names)})"
            else:
                info_msg = "All processing steps available"
            
            return False, initial_value, slider_max, info_msg, "step-info-text enabled"
            
        except Exception as e:
            print(f"[DEBUG] Error updating step slider state: {e}")
            return True, 5, 5, "Error checking processing steps", "step-info-text"

    @app.callback(
        [Output(f'step-label-{i}', 'className') for i in range(6)],
        [Input('processing-step-slider', 'value'),
         Input('processing-step-slider', 'disabled'),
         Input('selected-file-store', 'data'),
         Input('selected-dataset-store', 'data')],
        prevent_initial_call=False
    )
    def update_step_labels(step_value, is_disabled, selected_file_data, selected_dataset):
        """
        Update step label highlighting based on current slider value, disabled state, and available steps.
        """
        print(f"\n� UPDATE_STEP_LABELS called")
        print(f"   step_value={step_value}, is_disabled={is_disabled}")
        print(f"   file_data={selected_file_data}, dataset={selected_dataset}")
        
        if step_value is None:
            step_value = 0
        
        # Check for missing steps only when a file is selected
        if (selected_file_data and selected_dataset and 
            'UnifiedPreprocessed' in selected_dataset):
            try:
                filename = selected_file_data.get('filename', '')
                print(f"📁 Processing shape: {filename}")
                
                # Get the file data directly from the dataset
                file_df = get_cached_dataset_data(selected_dataset)
                if file_df is not None and not file_df.empty:
                    # Find the row with matching filename
                    matching_rows = file_df[file_df['filename'] == filename]
                    if not matching_rows.empty:
                        row = matching_rows.iloc[0]
                        print(f"✅ Found matching row for {filename}")
                    
                    # Get available steps for this shape
                    from core.file_index import get_available_steps
                    available_steps_info = get_available_steps(row)
                    available_steps = available_steps_info.get('available_step_indices', [])
                    print(f"� Available steps for {filename}: {available_steps}")
                    
                    class_names = []
                    for i in range(6):
                        if i not in available_steps:
                            # This step is missing
                            class_names.append("step-label missing")
                            print(f"🔴 Step {i} is MISSING")
                        elif i == step_value:
                            class_names.append("step-label active")
                            print(f"🟢 Step {i} is ACTIVE")
                        else:
                            class_names.append("step-label")
                            print(f"⚪ Step {i} is NORMAL")
                    
                    print(f"🎯 Final class names: {class_names}")
                    return class_names
                    
            except Exception as e:
                print(f"❌ Error in step label processing: {e}")
                import traceback
                traceback.print_exc()
        
        # Default fallback - no missing steps styling
        class_names = []
        for i in range(6):
            if is_disabled:
                class_names.append("step-label disabled")
            elif i == step_value:
                class_names.append("step-label active")
            else:
                class_names.append("step-label")
        
        print(f"📋 Default class names: {class_names}")
        return class_names

    @app.callback(
        Output('step-info-display', 'children', allow_duplicate=True),
        [Input('processing-step-slider', 'value')],
        prevent_initial_call=True
    )
    def update_step_info_display(step_value):
        """
        Update the step info display based on slider position.
        """
        if step_value is None:
            return "Select step"
        
        step_info = get_step_display_info(step_value)
        return f"Step {step_value}: {step_info['name']} - {step_info['description']}"

