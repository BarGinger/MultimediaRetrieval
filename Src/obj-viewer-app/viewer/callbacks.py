
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
from core.file_index import get_file_tree
from core.analysis_cache import merge_analysis_data, get_analysis_data
from core.shapeMesh import ShapeMesh

def fast_merge_analysis_data(file_df, dataset):
    """
    Fast version of merge_analysis_data that only uses cached data.
    Does not compute analysis on-the-fly to avoid slowdowns during dataset switching.
    """
    analysis_df = get_analysis_data(dataset)
    if analysis_df is not None:
        # Fast merge with cached data only
        file_df_copy = file_df.copy()
        file_df_copy['base_filename'] = file_df_copy['filename'].str.replace('_unified.obj', '.obj')
        analysis_df_copy = analysis_df.copy()
        analysis_df_copy['base_filename'] = analysis_df_copy['filename']
        
        merged = pd.merge(
            file_df_copy, 
            analysis_df_copy[['category', 'base_filename', 'num_vertices', 'num_faces']],
            on=['category', 'base_filename'], 
            how='left'
        ).drop('base_filename', axis=1)
        return merged
    else:
        # No cached data - add empty columns
        file_df = file_df.copy()
        file_df['num_vertices'] = None
        file_df['num_faces'] = None
        return file_df


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
         Output('sort-order', 'data-order')],
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
            return "↑", "Sort Order: Ascending (click to change to Descending)", "asc"
        else:
            print(f"✅ Creating descending sort")  # Debug
            return "↓", "Sort Order: Descending (click to change to Ascending)", "desc"

    # Show toast message immediately when sort button is clicked (SAME AS LOADING)
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
                if (toastBar && n_clicks % 2 === 0) {
                    document.getElementById('toast-icon').innerHTML = '↑';
                    document.getElementById('toast-message').innerHTML = 'Sort order changed to Ascending';
                    toastBar.style.display = 'block';
                } else if (toastBar) {
                    document.getElementById('toast-icon').innerHTML = '↓';
                    document.getElementById('toast-message').innerHTML = 'Sort order changed to Descending';
                    toastBar.style.display = 'block';
                }
            }, 10);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('toast-message-bar', 'id', allow_duplicate=True),
        Input('sort-order', 'n_clicks'),
        prevent_initial_call=True
    )

    # Auto-hide toast after 3 seconds - triggered by any button click
    app.clientside_callback(
        """
        function(sort_clicks, vertices_clicks, faces_clicks) {
            const ctx = window.dash_clientside.callback_context;
            if (!ctx.triggered.length) {
                return window.dash_clientside.no_update;
            }
            
            // Clear any existing timeout
            if (window.toastTimeout) {
                clearTimeout(window.toastTimeout);
            }
            
            // Set new timeout to hide after 3 seconds
            window.toastTimeout = setTimeout(function() {
                const toastBar = document.getElementById('toast-message-bar');
                if (toastBar) {
                    toastBar.style.display = 'none';
                }
            }, 3000);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('toast-message-bar', 'id', allow_duplicate=True),
        [Input('sort-order', 'n_clicks'),
         Input('avg-vertices-btn', 'n_clicks'),
         Input('avg-faces-btn', 'n_clicks')],
        prevent_initial_call=True
    )

    # Show toast for filename filter changes
    app.clientside_callback(
        """
        function(filename_filter) {
            try {
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
                    const toastIcon = document.getElementById('toast-icon');
                    const toastMessage = document.getElementById('toast-message');
                    
                    if (toastBar && toastIcon && toastMessage) {
                        toastIcon.innerHTML = '🔍';
                        toastMessage.innerHTML = 'Filename filter applied: ' + filename_filter;
                        toastBar.style.display = 'block';
                        
                        // Auto-hide after 2 seconds for filter
                        setTimeout(function() {
                            if (toastBar) {
                                toastBar.style.display = 'none';
                            }
                        }, 2000);
                    }
                }, 10);
                
                return window.dash_clientside.no_update;
            } catch (error) {
                console.error('Error in filename filter toast:', error);
                return window.dash_clientside.no_update;
            }
        }
        """,
        Output('toast-message-bar', 'id', allow_duplicate=True),
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
                }
            }, 10);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('toast-message-bar', 'id', allow_duplicate=True),
        Input('avg-faces-btn', 'n_clicks'),
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
         State('sort-field', 'value'),
         State('sort-order', 'data-order'),
         State('selected-dataset-store', 'data')],
        prevent_initial_call=True
    )
    def navigate_to_average(avg_vertices_clicks, avg_faces_clicks, selected_category, sort_field, sort_order, selected_dataset):
        """Navigate to the item closest to average vertices or faces"""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Get the current file list using the same logic as the main callback
        try:
            file_df = get_file_tree(selected_dataset)
        except Exception as e:
            print(f"Error loading dataset {selected_dataset}: {e}")
            return no_update, no_update, no_update
        
        # Always try to merge cached analysis data
        analysis_df = get_analysis_data(selected_dataset)
        if analysis_df is not None:
            file_df_copy = file_df.copy()
            file_df_copy['base_filename'] = file_df_copy['filename'].str.replace('_unified.obj', '.obj')
            analysis_df_copy = analysis_df.copy()
            analysis_df_copy['base_filename'] = analysis_df_copy['filename']
            
            file_df = pd.merge(
                file_df_copy, 
                analysis_df_copy[['category', 'base_filename', 'num_vertices', 'num_faces']],
                on=['category', 'base_filename'], 
                how='left'
            ).drop('base_filename', axis=1)
        else:
            # No analysis data available, can't find average
            print(f"⚠️ No cached analysis for {selected_dataset} - cannot find average")
            toast_data = create_toast_data("No analysis data available for average calculation", "warning", "⚠️")
            return no_update, no_update, toast_data
        
        # Apply category filter
        df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
        
        # Apply sorting
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
                return selected_idx, info, dash.no_update
            except Exception as e:
                err_info = html.Div([
                    html.H4("❌ Error Loading Average Shape", style={'color': '#e74c3c', 'marginBottom': '15px'}),
                    html.Div([html.Strong("📄 File: "), row['filepath']], style={'marginBottom': '8px'}),
                    html.Div([html.Strong("⚠️ Error: "), str(e)], style={'color': '#e74c3c'})
                ])
                error_toast_data = dash.no_update  # No old toast system
                return selected_idx, err_info, error_toast_data
        
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

    # Client-side callback to scroll to selected file in the list
    app.clientside_callback(
        """
        function(selected_idx) {
            console.log('Client-side scroll callback triggered with selected_idx:', selected_idx);
            
            if (selected_idx === null || selected_idx === undefined) {
                console.log('No selection, returning no_update');
                return window.dash_clientside.no_update;
            }
            
            // Wait a bit for the DOM to be ready
            setTimeout(function() {
                // Find the file list container - try multiple selectors
                let fileListContainer = document.querySelector('#file-list .file-list-panel');
                if (!fileListContainer) {
                    fileListContainer = document.querySelector('#file-list > div');
                }
                if (!fileListContainer) {
                    fileListContainer = document.querySelector('.file-list-panel');
                }
                
                if (!fileListContainer) {
                    console.log('File list container not found');
                    return;
                }
                console.log('Found file list container:', fileListContainer);
                
                // Find all file buttons - try multiple selectors
                let fileButtons = fileListContainer.querySelectorAll('button.file-button');
                if (fileButtons.length === 0) {
                    fileButtons = fileListContainer.querySelectorAll('button[id*="file-btn"]');
                }
                if (fileButtons.length === 0) {
                    fileButtons = fileListContainer.querySelectorAll('button');
                }
                
                console.log('Found', fileButtons.length, 'file buttons, targeting index', selected_idx);
                
                if (fileButtons.length > selected_idx) {
                    const targetButton = fileButtons[selected_idx];
                    console.log('Target button found:', targetButton);
                    
                    // Remove any existing selection highlights
                    fileButtons.forEach(btn => {
                        btn.classList.remove('selected-file');
                        btn.style.backgroundColor = '';
                        btn.style.borderColor = '';
                        btn.style.color = '';
                    });
                    
                    // Add selection styling to target button
                    targetButton.classList.add('selected-file');
                    targetButton.style.backgroundColor = '#3498db';
                    targetButton.style.borderColor = '#2980b9';
                    targetButton.style.color = '#ffffff';
                    targetButton.style.transition = 'all 0.3s ease';
                    
                    // Scroll to the target button within the container
                    const containerRect = fileListContainer.getBoundingClientRect();
                    const buttonRect = targetButton.getBoundingClientRect();
                    
                    // Calculate if button is visible in container
                    const isVisible = (
                        buttonRect.top >= containerRect.top &&
                        buttonRect.bottom <= containerRect.bottom
                    );
                    
                    if (!isVisible) {
                        // Scroll the button into view within the container
                        const scrollTop = targetButton.offsetTop - fileListContainer.offsetTop - 
                                        (fileListContainer.clientHeight / 2) + (targetButton.clientHeight / 2);
                        
                        fileListContainer.scrollTo({
                            top: scrollTop,
                            behavior: 'smooth'
                        });
                    }
                    
                    console.log('Selection and scroll applied to target button');
                } else {
                    console.log('Target index', selected_idx, 'is out of range for', fileButtons.length, 'buttons');
                }
            }, 100); // Small delay to ensure DOM is ready
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('file-list', 'id'),  # Dummy output
        Input('selected-file-store', 'data'),
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
                        toast.classList.add('show');
                    }, index * 100); // Stagger animations
                });
                
                // Auto-hide after 3 seconds
                setTimeout(function() {
                    toastElements.forEach(function(toast) {
                        toast.classList.remove('show');
                        setTimeout(function() {
                            if (toast.parentNode) {
                                toast.parentNode.removeChild(toast);
                            }
                        }, 300); // Wait for fade out animation
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

    # Toast system using stores (no DOM conflicts)
    @app.callback(
        [Output('toast-container', 'children'),
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
            return [], True, 0
        
        print(f"✅ Creating toast: {toast_data['message']}")  # Debug
        toast_element = html.Div([
            html.Span(toast_data['icon'], className="toast-icon"),
            html.Span(toast_data['message'], className="toast-message")
        ], className=f"toast {toast_data['type']}")
        
        return [toast_element], False, 0  # Enable interval and reset counter

    @app.callback(
        [Output('toast-container', 'children', allow_duplicate=True),
         Output('toast-interval', 'disabled', allow_duplicate=True)],
        Input('toast-interval', 'n_intervals'),
        State('toast-interval', 'disabled'),
        prevent_initial_call=True
    )
    def clear_toast_after_delay(n_intervals, interval_disabled):
        """Clear toast after 40 intervals (4 seconds at 100ms)"""
        if interval_disabled:
            return no_update, no_update
        
        if n_intervals >= 40:  # 4 seconds
            return [], True  # Clear toast and disable interval
        
        return no_update, no_update

    # 1) File list render
    @app.callback(
        Output('file-list', 'children'),
        [Input('category-filter', 'value'),
         Input('filename-filter', 'value'),
         Input('vertices-operator', 'value'),
         Input('vertices-value', 'value'),
         Input('faces-operator', 'value'),
         Input('faces-value', 'value'),
         Input('sort-field', 'value'),
         Input('sort-order', 'data-order'),
         Input('selected-dataset-store', 'data')]
    )
    def update_file_list(selected_category, filename_filter, vertices_op, vertices_val, faces_op, faces_val, sort_field, sort_order, selected_dataset):        
        """
        Render the list of files based on current filters and sorting.
        Optimized to avoid slow analysis computation during dataset switching.
        """
        return update_file_list_internal('all', selected_category, filename_filter, vertices_op, vertices_val, faces_op, faces_val, sort_field, sort_order, selected_dataset)

    def update_file_list_internal(avg_filter, selected_category, filename_filter, vertices_op, vertices_val, faces_op, faces_val, sort_field, sort_order, selected_dataset):        
        """
        Render the list of files based on current filters and sorting.
        Optimized to avoid slow analysis computation during dataset switching.

        Parameters:
        - avg_filter: str, average filter option ('none', 'avg_faces', 'avg_vertices')
        - selected_category: str, selected category filter ('all' or specific category)
        - filename_filter: str, filename pattern filter (supports wildcards like m*, *153*)
        - vertices_op: str, vertices comparison operator ('eq', 'gt', 'lt')
        - vertices_val: int/None, vertices value for comparison
        - faces_op: str, faces comparison operator ('eq', 'gt', 'lt')
        - faces_val: int/None, faces value for comparison
        - sort_field: str, field to sort by ('category', 'num_vertices', 'num_faces')
        - sort_order: str, sort order ('asc' or 'desc')
        - selected_dataset: str, currently selected dataset from dropdown

        Returns:
        - List of HTML button elements representing the files
        """
        if selected_dataset is None or selected_dataset == "":
            selected_dataset = 'Data'

        file_df = get_file_tree(selected_dataset)

        if file_df.empty:
            return [html.P("❌ No files found in Data directory",
                           style={'color': 'red', 'textAlign': 'center'})]

        # **OPTIMIZATION**: Always try to show cached analysis data (fast), only skip slow computation
        # Try to get cached analysis first (fast)
        analysis_df = get_analysis_data(selected_dataset)
        if analysis_df is not None:
            # Merge with cached data (fast)
            file_df_copy = file_df.copy()
            file_df_copy['base_filename'] = file_df_copy['filename'].str.replace('_unified.obj', '.obj')
            analysis_df_copy = analysis_df.copy()
            analysis_df_copy['base_filename'] = analysis_df_copy['filename']
            
            file_df = pd.merge(
                file_df_copy, 
                analysis_df_copy[['category', 'base_filename', 'num_vertices', 'num_faces']],
                on=['category', 'base_filename'], 
                how='left'
            ).drop('base_filename', axis=1)
            print(f"✅ Merged cached analysis for {selected_dataset}")
        else:
            # No cached analysis - add empty columns and defer computation to file clicks
            print(f"⚠️ No cached analysis for {selected_dataset} - analysis will be computed when files are clicked")
            file_df['num_vertices'] = None
            file_df['num_faces'] = None
        
        # Check if we need analysis for sorting/filtering operations
        needs_analysis_ops = (sort_field in ['num_vertices', 'num_faces'] or 
                             avg_filter in ['avg_faces', 'avg_vertices'] or
                             (vertices_val is not None and vertices_val != '') or
                             (faces_val is not None and faces_val != ''))
        
        # If we need analysis for operations but don't have cached data, show a warning
        if needs_analysis_ops and analysis_df is None:
            print(f"⚠️ Cannot perform {sort_field or avg_filter} operation - no cached analysis data available")

        df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
        
        # Apply filename filtering if provided
        if filename_filter and filename_filter.strip() and not df.empty and 'filename' in df.columns:
            try:
                import fnmatch
                pattern = filename_filter.strip()
                # Use fnmatch to filter filenames with wildcard support
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
            if file_idx >= len(df):
                return no_update, no_update
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
         Input('normalization-toggle', 'value'),
         Input('selected-dataset-store', 'data')],
        [State('category-filter', 'value'),
         State('sort-field', 'value'),
         State('sort-order', 'data-order'),
         Input('3d-plot', 'figure')],
        prevent_initial_call=True
    )
    def update_plot(display_options, 
                    selected_file_idx, 
                    mesh_color,                      
                    show_normalized,
                    selected_dataset,
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
        # Use fast merge to avoid slowdowns during dataset switching
        file_df = fast_merge_analysis_data(file_df, selected_dataset)

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
        
        # Now use the filtered/sorted DataFrame for index lookup
        if selected_file_idx >= len(df):
            return create_3d_plot(np.array([]), np.array([]), "Select a shape to view",
                                  mesh_color=mesh_color or 'lightblue')
        row = df.iloc[selected_file_idx]
        
        # Create ShapeMesh instance and handle special cases for different datasets
        try:
            # Special handling for NormalizedShapes dataset
            if selected_dataset == 'NormalizedShapes':
                # NormalizedShapes dataset contains pre-normalized files
                mesh = ShapeMesh.from_file_row(row)
                vertices = mesh.vertices  # Already normalized
                title_suffix = " (Pre-normalized Dataset)"
                camera_config = None  # Use default camera for normalized shapes
                print(f"[DEBUG] Using pre-normalized vertices from NormalizedShapes dataset for {row['filename']}")
            
            # Handle normalization toggle for other datasets
            elif show_normalized and 'normalized' in show_normalized:
                from core.normalized_cache import normalized_cache
                # Try to load from cache first
                if normalized_cache.is_normalized_available(row['filename'], selected_dataset):
                    mesh = normalized_cache.load_normalized_shape(row['filename'], selected_dataset)
                    vertices = mesh.vertices
                    title_suffix = " (Cached Normalized)"
                    print(f"[DEBUG] Using cached normalized vertices for {row['filename']}")
                else:
                    # Fall back to computing normalization
                    mesh = ShapeMesh.from_file_row(row)
                    vertices = mesh.apply_full_normalization()
                    title_suffix = " (Computed Normalized)"
                    print(f"[DEBUG] Computing normalized vertices for {row['filename']} (cache not available)")
                
                camera_config = None  # Use default camera for normalized shapes
            else:
                # Original shape
                mesh = ShapeMesh.from_file_row(row)
                vertices = mesh.vertices
                title_suffix = ""
                camera_config = mesh.get_optimal_camera_position()
                print(f"[DEBUG] Using original vertices for {row['filename']}")
            
            faces = mesh.faces
                
        except Exception as e:
            print(f"[DEBUG] ShapeMesh failed: {e}")
            # Fallback to original method if ShapeMesh fails
            vertices, faces = OBJParser.parse_obj_file(row['filepath'])
            camera_config = None
            title_suffix = ""
        
        show_wire = 'wireframe' in (display_options or [])
        title = f"{row['category']} - {row['filename']}{title_suffix}"

        fig = create_3d_plot(vertices, faces, title, show_wireframe=show_wire,
                              mesh_color=mesh_color or 'lightblue',
                              smooth_shading=smooth_shading,
                              camera_config=camera_config)
        
        # If user had a previous camera position, restore it
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
    def update_normalization_toggle(selected_file_idx, selected_dataset, selected_category, sort_field, sort_order):
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
        if selected_file_idx is None:
            return options, []
        
        # Get the current file data to check filename
        if selected_dataset is None or selected_dataset == "":
            selected_dataset = 'Data'
        
        try:
            file_df = get_file_tree(selected_dataset)
            file_df = fast_merge_analysis_data(file_df, selected_dataset)
            
            if file_df.empty:
                return options, []
            
            # Apply same filtering and sorting logic as other callbacks
            df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
            ascending = True if sort_order == 'asc' else False
            df = df.copy()
            
            if sort_field == 'category':
                df = df.sort_values(by=['category', 'filename'], ascending=ascending)
            elif sort_field in ['num_vertices', 'num_faces'] and sort_field in df.columns:
                df[sort_field] = df[sort_field].fillna(0)
                df = df.sort_values(by=sort_field, ascending=ascending)
            
            df = df.reset_index(drop=True)
            
            # Check if the selected file index is valid
            if selected_file_idx >= len(df):
                return options, []
            
            # Get the filename and check for _normalized suffix
            row = df.iloc[selected_file_idx]
            filename = row['filename']
            
            # Check if filename contains '_normalized' (case-insensitive)
            if '_normalized' in filename.lower() or '_unified' in filename.lower():
                # For normalized files, disable the checkbox and check it
                options = [{'label': '', 'value': 'normalized', 'disabled': True}]
                return options, ['normalized']  # Check the normalization toggle
            else:
                # For non-normalized files, disable the checkbox and uncheck it
                options = [{'label': '', 'value': 'normalized', 'disabled': True}]
                return options, []  # Uncheck the normalization toggle
                
        except Exception as e:
            print(f"[DEBUG] Error updating normalization toggle: {e}")
            return options, []

