from dash import dcc, html, Input, Output, no_update, State, callback_context
import dash
import numpy as np
import os
import pandas as pd
import json
import uuid
import time
import re
import math
import colorsys
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from core.dataset_cache import get_cached_dataset_data
from core.obj_parser import OBJParser
from core.shapeMesh import ShapeMesh
from core.file_index import get_step_file_path
from core.file_index import get_available_steps, get_step_display_info
from core.plotting import create_3d_plot
from .category_colors import CATEGORIES_LIST, CATEGORY_COLOR_MAP

# Global cache for distance matrix (loaded once at startup)
_DISTANCE_MATRIX_CACHE = None
_DISTANCE_MATRIX_PATH = None

# Global cache for t-SNE embedding (loaded once at startup)
_TSNE_EMBEDDING_CACHE = None
_TSNE_LABELS_CACHE = None

def get_cached_distance_matrix():
    """Load and cache the distance matrix in memory for fast KNN queries.
    
    Returns the distance matrix DataFrame or None if not available.
    Caches result globally to avoid reloading CSV on every query.
    """
    global _DISTANCE_MATRIX_CACHE, _DISTANCE_MATRIX_PATH
    
    # Return cached matrix if already loaded
    if _DISTANCE_MATRIX_CACHE is not None:
        return _DISTANCE_MATRIX_CACHE
    
    # Determine path to distance matrix
    base_dir = os.path.dirname(os.path.abspath(__file__))
    distance_path = os.path.join(base_dir, "..", "..", "matching", "matrix_rank_based_optimized.csv")
    distance_path = os.path.normpath(distance_path)
    
    # Check if file exists
    if not os.path.exists(distance_path):
        print(f"⚠️ Distance matrix not found at: {distance_path}")
        print(f"   KNN retrieval will not be available.")
        return None
    
    try:
        print(f"📊 Loading distance matrix from: {distance_path}")
        # Load matrix with first column as index
        df = pd.read_csv(distance_path, index_col=0)
        print(f"✅ Distance matrix loaded: {df.shape[0]} × {df.shape[1]} = {df.shape[0] * df.shape[1]:,} distances")
        
        # Cache for future use
        _DISTANCE_MATRIX_CACHE = df
        _DISTANCE_MATRIX_PATH = distance_path
        
        return df
    except Exception as e:
        print(f"❌ Error loading distance matrix: {e}")
        return None


def get_cached_tsne_data():
    """Load and cache the t-SNE embedding and class labels for clustering visualization.
    
    Returns tuple: (embedding_df, labels_df) or (None, None) if not available.
    Caches result globally to avoid reloading CSVs on every modal open.
    """
    global _TSNE_EMBEDDING_CACHE, _TSNE_LABELS_CACHE
    
    # Return cached data if already loaded
    if _TSNE_EMBEDDING_CACHE is not None and _TSNE_LABELS_CACHE is not None:
        return _TSNE_EMBEDDING_CACHE, _TSNE_LABELS_CACHE
    
    # Determine paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scalability_dir = os.path.join(base_dir, "..", "..", "scalability")
    embedding_path = os.path.join(scalability_dir, "topology_graph.csv")
    labels_path = os.path.join(scalability_dir, "class_labels.csv")
    
    embedding_path = os.path.normpath(embedding_path)
    labels_path = os.path.normpath(labels_path)
    
    # Check if files exist
    if not os.path.exists(embedding_path):
        print(f"⚠️ t-SNE embedding not found at: {embedding_path}")
        return None, None
    
    if not os.path.exists(labels_path):
        print(f"⚠️ Class labels not found at: {labels_path}")
        return None, None
    
    try:
        print(f"📊 Loading t-SNE embedding from: {embedding_path}")
        embedding_df = pd.read_csv(embedding_path, header=0, index_col=0)
        print(f"✅ t-SNE embedding loaded: {len(embedding_df)} points")
        
        print(f"📊 Loading class labels from: {labels_path}")
        labels_df = pd.read_csv(labels_path, header=0, index_col=0)
        
        # Ensure 'shape' column exists
        if "shape" not in labels_df.columns:
            labels_df = labels_df.reset_index().rename(columns={"index": "shape"})
        
        print(f"✅ Class labels loaded: {len(labels_df)} shapes")
        
        # Cache for future use
        _TSNE_EMBEDDING_CACHE = embedding_df
        _TSNE_LABELS_CACHE = labels_df
        
        return embedding_df, labels_df
    except Exception as e:
        print(f"❌ Error loading t-SNE data: {e}")
        return None, None


def _parse_hist_and_bins(hist, bins=None):
    """Parse histogram and bins stored in various formats into (mids, values).

    Accepts lists, numpy arrays, JSON strings, or Python reprs. Returns (mids, vals)
    where mids is list of bin centers (or 0..N-1 if bins missing) and vals is list of numbers.
    """
    import ast
    try:
        # None guard
        if hist is None:
            return None, None

        # If it's a pandas Series or numpy array
        if hasattr(hist, 'tolist'):
            vals = list(hist.tolist())
        elif isinstance(hist, (list, tuple)):
            vals = list(hist)
        elif isinstance(hist, str):
            # Try JSON first
            try:
                parsed = json.loads(hist)
                vals = list(parsed)
            except Exception:
                try:
                    vals = list(ast.literal_eval(hist))
                except Exception:
                    # as last resort, try comma-split
                    vals = [float(x.strip()) for x in hist.split(',') if x.strip()]
        else:
            # Fallback to single numeric
            vals = [float(hist)]

        # parse bins similarly
        bin_edges = None
        if bins is not None:
            if hasattr(bins, 'tolist'):
                bin_edges = list(bins.tolist())
            elif isinstance(bins, (list, tuple)):
                bin_edges = list(bins)
            elif isinstance(bins, str):
                try:
                    parsed = json.loads(bins)
                    bin_edges = list(parsed)
                except Exception:
                    try:
                        bin_edges = list(ast.literal_eval(bins))
                    except Exception:
                        bin_edges = None

        # compute mids
        if bin_edges and len(bin_edges) >= 2:
            # If edges length == vals length, assume they are centers already
            if len(bin_edges) == len(vals):
                mids = bin_edges
            elif len(bin_edges) == len(vals) + 1:
                mids = [(bin_edges[i] + bin_edges[i+1]) / 2.0 for i in range(len(bin_edges)-1)]
            else:
                # fallback to 0..N-1
                mids = list(range(len(vals)))
        else:
            mids = list(range(len(vals)))

        # Ensure floats (be forgiving for NaN/None/strings)
        def _to_float_list(seq):
            out = []
            for x in seq:
                try:
                    # pandas NA guard
                    if pd.isna(x):
                        out.append(0.0)
                    else:
                        out.append(float(x))
                except Exception:
                    try:
                        out.append(float(str(x)))
                    except Exception:
                        out.append(0.0)
            return out

        mids = _to_float_list(mids)
        vals = _to_float_list(vals)
        return mids, vals
    except Exception:
        return None, None
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
    """Register all Dash callbacks using the provided app and dataset info.

    This function was unintentionally removed; restore it so callers can import
    and register callbacks by passing the Dash `app` instance.
    """

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
        [Output('dataset-selector', 'value', allow_duplicate=True),
         Output('selected-dataset-store', 'data', allow_duplicate=True),
         Output('category-filter', 'value', allow_duplicate=True),
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
        """Reset all filters to their default values, including dataset selection.

        Uses the outer-scope `default_dataset` captured when `register_callbacks` was called so
        the UI resets to the configured app default.
        """
        if n_clicks and n_clicks > 0:
            # Reset dataset selector and stored dataset to the default value
            ds = default_dataset if default_dataset else dash.no_update
            # Return order must match Outputs: dataset-selector, selected-dataset-store, category, filename, v-op, v-val, f-op, f-val, sort-field
            return ds, ds, 'all', '', 'gt', '', 'gt', '', 'category'
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update)

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

    # NOTE: aux-plots-message removed; spinner-only loading remains for similar shapes

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
                    html.Span(f"🔺 Vertices: {vertices_count}", className="stats-text", style={'fontSize': '0.75em', 'color': '#888'}),
                    html.Span(f"🔷 Faces: {faces_count}", className="stats-text", style={'fontSize': '0.75em', 'color': '#888', 'marginLeft': '8px'})
                ])
            ]),
            id={'type': 'file-btn', 'filename': encoded_filename},
            className='file-button',
            n_clicks=0,
            **{'data-filename': item['filename'], 'data-file-index': item.get('original_index', 0), 'data-category': item.get('category', '')}
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
        [Output('shape-info', 'children', allow_duplicate=True),
         Output('selected-file-store', 'data', allow_duplicate=True)],
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

    # Callback to update Vertices and Faces count based on slider step and selected dataset
    @app.callback(
        [Output('shape-vertices', 'children'), Output('shape-faces', 'children')],
        [Input('processing-step-slider', 'value'), Input('dataset-selector', 'value')],
        prevent_initial_call=True
    )
    def update_shape_info(step, dataset):
        """Update only the vertices and faces numeric spans when the slider changes for UnifiedPreprocessed/Data.

        This avoids overwriting the entire `shape-info` card produced by other callbacks.
        """
        analysis_results_path = "Datasets/UnifiedPreprocessed/Data/analysis_results_unifiedpreprocessed_data.csv"
        # fallback to the correctly named file if present
        alt_path = "Datasets/UnifiedPreprocessed/Data/analysis_results_unifiedPreprocessed_data.csv"
        if os.path.exists(analysis_results_path):
            path = analysis_results_path
        elif os.path.exists(alt_path):
            path = alt_path
        else:
            return dash.no_update, dash.no_update

        try:
            analysis_df = pd.read_csv(path)
        except Exception as e:
            print(f"[update_shape_info] Failed to read analysis CSV '{path}': {e}")
            return dash.no_update, dash.no_update

        if dataset != "UnifiedPreprocessed/Data" or step is None or analysis_df.empty:
            return dash.no_update, dash.no_update

        # Normalize column names (case-insensitive) and map common variants
        col_map = {c.lower(): c for c in analysis_df.columns}

        # Common alternative names for logical fields in analysis CSVs
        candidates = {
            'step': ['step', 'processing_step', 'step_id', 'shape_file', 'filename'],
            'vertices': ['num_vertices', 'vertices', 'verts', 'vertex_count'],
            'faces': ['num_faces', 'faces', 'face_count']
        }

        found = {}
        for logical, names in candidates.items():
            for n in names:
                if n in col_map:
                    found[logical] = col_map[n]
                    break

        # If any logical column is missing, log and bail out gracefully
        missing = [k for k in candidates.keys() if k not in found]
        if missing:
            print(f"[update_shape_info] Analysis CSV missing logical columns: {missing}. Available: {list(analysis_df.columns)}")
            return dash.no_update, dash.no_update

        step_col = found['step']
        vert_col = found['vertices']
        face_col = found['faces']

        # Safely compare step values (allow numeric/string mismatch)
        # If the step column contains filenames (e.g., 'shape_file' or 'filename'), match by the expected step suffix
        step_col_lower = step_col.lower() if isinstance(step_col, str) else ''
        if any(k in step_col_lower for k in ('shape', 'file', 'filename')):
            # Expected processing step suffixes used in filenames
            expected_steps = [
                "00_original",
                "01_remeshed",
                "02_translated",
                "03_aligned",
                "04_flipped",
                "05_scaled",
                "06_fill_holes_and_orientation"
            ]
            try:
                idx = int(step)
                if 0 <= idx < len(expected_steps):
                    suffix = expected_steps[idx]
                    mask = analysis_df[step_col].astype(str).str.contains(rf"_{re.escape(suffix)}\.obj$", regex=True, na=False)
                else:
                    # Out-of-range numeric step: fallback to substring match
                    mask = analysis_df[step_col].astype(str).str.contains(re.escape(str(step)), regex=True, na=False)
            except Exception:
                s = str(step)
                if s in expected_steps:
                    mask = analysis_df[step_col].astype(str).str.contains(rf"_{re.escape(s)}\.obj$", regex=True, na=False)
                else:
                    # Fallback: check if the filename contains the provided string
                    mask = analysis_df[step_col].astype(str).str.contains(re.escape(s), regex=True, na=False)
        else:
            try:
                # Try numeric comparison first
                step_val = int(step)
                mask = pd.to_numeric(analysis_df[step_col], errors='coerce') == step_val
            except Exception:
                # Fallback to string comparison
                mask = analysis_df[step_col].astype(str) == str(step)

        current_file_data = analysis_df[mask]
        if current_file_data.empty:
            print(f"[update_shape_info] No row for step={step} in '{path}'")
            return dash.no_update, dash.no_update

        vertices_count = current_file_data[vert_col].iloc[0]
        faces_count = current_file_data[face_col].iloc[0]

        # Format with commas like the rest of the UI
        try:
            vertices_str = f"{int(vertices_count):,}"
        except Exception:
            vertices_str = str(vertices_count)
        try:
            faces_str = f"{int(faces_count):,}"
        except Exception:
            faces_str = str(faces_count)

        return vertices_str, faces_str

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
         State('3d-plot', 'figure')], # Changed to State
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
        Update the 3D plot based on user selections.
        - Preserves camera view when changing steps/display options for the same shape.
        - Resets camera view when a new shape is selected.
        """
        ctx = dash.callback_context
        triggered_id = ""
        if ctx.triggered:
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]


        # Ensure camera_config, vertices, and faces are always defined
        camera_config = None
        vertices = np.array([])
        faces = np.array([])


        # Always try to extract the current camera from the figure if available
        prev_camera = None
        prev_filename = None
        if current_fig and 'layout' in current_fig and 'scene' in current_fig['layout']:
            prev_camera = current_fig['layout']['scene'].get('camera', None)
            # Try to extract previous filename from figure layout title (if present)
            prev_title = current_fig['layout'].get('title', {}).get('text', '')
            if '-' in prev_title:
                prev_filename = prev_title.split('-')[-1].strip().split()[0]

        # Determine if the selected shape has changed (reset camera if so)
        selected_filename = None
        if isinstance(selected_file_data, dict):
            selected_filename = selected_file_data.get('filename')

        reset_camera = (prev_filename is not None and selected_filename is not None and prev_filename != selected_filename)

        # Camera to use: reset if new shape, else preserve
        camera = None if reset_camera else prev_camera

        smooth_shading = 'smooth_shading' in (display_options or [])

        if selected_file_data is None:
            return create_3d_plot(np.array([]), np.array([]), "Select a shape to view",
                                  mesh_color=mesh_color or 'lightblue'), no_update, no_update

        if not isinstance(selected_file_data, dict):
            print(f"❌ ERROR: selected_file_data should be dict, got {type(selected_file_data)}: {selected_file_data}")
            return create_3d_plot(np.array([]), np.array([]), "Invalid selection data",
                                  mesh_color=mesh_color or 'lightblue'), no_update, no_update

        selected_filename = selected_file_data.get('filename')
        file_dataset = selected_file_data.get('dataset', selected_dataset)

        if not selected_filename:
            return create_3d_plot(np.array([]), np.array([]), "No valid shape selected",
                                  mesh_color=mesh_color or 'lightblue'), no_update, no_update

        if file_dataset is None or file_dataset == "":
            file_dataset = 'Data'

        file_df = get_cached_dataset_data(file_dataset)

        if file_df is None or file_df.empty:
            return create_3d_plot(np.array([]), np.array([]), "No valid shape selected",
                                  mesh_color=mesh_color or 'lightblue'), no_update, no_update

        # Ensure all step files are included in the merge
        if selected_dataset and 'UnifiedPreprocessed' in selected_dataset:
            file_df = get_cached_dataset_data(selected_dataset)
            if file_df is not None and not file_df.empty:
                # Filter rows to include all steps
                step_files = file_df[file_df['filename'].str.contains('_step')]
                if not step_files.empty:
                    file_df = pd.concat([file_df, step_files]).drop_duplicates()

        # Proceed with existing logic to find matching rows
        matching_rows = file_df[file_df['filename'] == selected_filename]
        if matching_rows.empty:
            return create_3d_plot(np.array([]), np.array([]), f"File {selected_filename} not found",
                                  mesh_color=mesh_color or 'lightblue'), no_update, no_update

        row = matching_rows.iloc[0]
        print(f"🎯 3D Plot: Loading {selected_filename} from {file_dataset}")

        step_row = row
        title_suffix = ""
        step_fallback_info = None

        try:
            # If processing steps are available and selected, use step logic
            if (processing_step is not None and
                selected_dataset and
                ('UnifiedPreprocessed' in selected_dataset or 'Normalized' in selected_dataset) and
                row.get('has_processing_steps', False)):

                if 'D00355' in row.get('filename', '') and processing_step == 1:
                    print(f"[DEBUG] D00355 step 1 requested - forcing fallback to step 0 (original)")
                    actual_file_path = row['filepath']
                    step_fallback_info = {
                        'requested_step': 1, 'actual_step': 0,
                        'requested_step_name': 'Remeshed', 'actual_step_name': 'Original',
                        'step_available': False
                    }
                    title_suffix = f" (Original Step - Fallback)"
                    step_row = row.copy()
                    step_row['filepath'] = actual_file_path
                else:
                    actual_file_path, actual_step_index, step_info = get_step_file_path(row, processing_step)
                    title_suffix = f" ({step_info['name']} Step)"
                    if step_info.get('fallback_used', False):
                        step_fallback_info = {
                            'requested_step': step_info['requested_step'],
                            'actual_step': step_info['actual_step'],
                            'requested_step_name': step_info.get('requested_step_name', 'Unknown'),
                            'actual_step_name': step_info['name'],
                            'step_available': step_info.get('step_available', False)
                        }
                        title_suffix = f" ({step_info['name']} Step - Fallback)"
                    step_row = row.copy()
                    step_row['filepath'] = actual_file_path
            # Always load mesh for the selected file (step_row is set above if needed)
            if selected_dataset == 'NormalizedShapes':
                mesh = ShapeMesh.from_file_row(step_row)
                vertices = mesh.vertices
                title_suffix += " (Pre-normalized Dataset)"
            elif show_normalized and 'normalized' in show_normalized and not row.get('has_processing_steps', False):
                from core.normalized_cache import normalized_cache
                if normalized_cache.is_normalized_available(row['filename'], selected_dataset):
                    mesh = normalized_cache.load_normalized_shape(row['filename'], selected_dataset)
                    vertices = mesh.vertices
                    title_suffix += " (Cached Normalized)"
                else:
                    mesh = ShapeMesh.from_file_row(step_row)
                    vertices = mesh.apply_full_normalization()
                    title_suffix += " (Computed Normalized)"
            else:
                mesh = ShapeMesh.from_file_row(step_row)
                vertices = mesh.vertices
                camera_config = mesh.get_optimal_camera_position()

            faces = mesh.faces

        except Exception as e:
            print(f"[DEBUG] ShapeMesh failed: {e}")
            file_path_to_use = step_row['filepath'] if 'step_row' in locals() else row['filepath']
            vertices, faces = OBJParser.parse_obj_file(file_path_to_use)
            title_suffix += " (Fallback Parser)"

        show_wire = 'wireframe' in (display_options or [])
        title = f"{row['category']} - {row['filename']}{title_suffix}"

        # Use preserved camera if available, otherwise use the calculated optimal one
        # Always pass a Plotly-style camera dict to create_3d_plot
        final_camera = camera if camera is not None else camera_config

        fig = create_3d_plot(vertices, faces, title, show_wireframe=show_wire,
                              mesh_color=mesh_color or 'lightblue',
                              smooth_shading=smooth_shading,
                              camera_config=final_camera)
        
        # Create toast notification for step fallback if needed
        # Handle missing step notification
        regular_toast_data = no_update  # Don't interfere with other toasts
        step_toast_data = no_update     # Don't send empty step toasts
        
        if step_fallback_info:
            if not step_fallback_info['step_available']:
                # Send step missing messages to the step-toast-store (positioned over 3D viewer)
                step_toast_data = create_toast_data(
                    f"ℹ️ Step '{step_fallback_info['requested_step_name']}' is not available. "
                    f"Displaying '{step_fallback_info['actual_step_name']}' instead.",
                    "info", "ℹ️"
                )

        return fig, regular_toast_data, step_toast_data
    

    # Similar shapes: sample random shapes from the (possibly selected) dataset
    def retrieve_random_shapes(selected_file_data, n):
        """Return up to n random rows (as dicts) from the selected dataset excluding the current file."""
        try:
            # Determine dataset to use
            dataset = None
            selected_filename = None
            if isinstance(selected_file_data, dict):
                selected_filename = selected_file_data.get('filename')
                dataset = selected_file_data.get('dataset')

            # Load dataset dataframe (fallback to closure file_df if dataset not provided)
            df_candidates = None
            if dataset:
                try:
                    df_candidates = get_cached_dataset_data(dataset)
                except Exception:
                    df_candidates = None

            if df_candidates is None or df_candidates.empty:
                df_candidates = file_df

            if selected_filename:
                df_candidates = df_candidates[df_candidates['filename'] != selected_filename]

            if df_candidates is None or df_candidates.empty:
                return []

            n = int(n or 5)
            if n <= 0:
                return []

            if n >= len(df_candidates):
                sampled = df_candidates.sample(n=len(df_candidates))
            else:
                sampled = df_candidates.sample(n=n)

            return sampled.to_dict('records')
        except Exception:
            return []
        
    def retrieve_closest_shapes(selected_file_data, n):
        """
        Return up to n closest shapes (as dicts) from the same dataset,
        using the precomputed distance matrix with optimized KNN.
        
        Optimizations:
        - Distance matrix cached in memory (loaded once)
        - Uses numpy for fast sorting
        - Efficient ID-based lookup
        
        Each dict includes all dataset info + 'distance' field.
        """
        try:
            # --- Validate input ---
            if not isinstance(selected_file_data, dict):
                return []
            selected_filename = selected_file_data.get('filename')
            dataset = selected_file_data.get('dataset')
            if not selected_filename:
                return []

            # --- Load the cached distance matrix (fast!) ---
            distance_matrix = get_cached_distance_matrix()
            if distance_matrix is None:
                print(f"⚠️ Distance matrix not available - falling back to random sampling")
                return retrieve_random_shapes(selected_file_data, n)

            # --- Match query shape by ID prefix ---
            m = re.match(r"([A-Za-z0-9]+)_", selected_filename)
            if not m:
                print(f"⚠️ Could not extract ID prefix from {selected_filename}")
                return []
            shape_id = m.group(1)

            # Find matching row in distance matrix (any processing step of this shape)
            matching_rows = [idx for idx in distance_matrix.index if idx.startswith(shape_id + "_")]
            if not matching_rows:
                print(f"⚠️ No matching row found for ID {shape_id} in distance matrix.")
                return []
            
            # Use first matching row (typically the _06 or _unified version)
            row_name = matching_rows[0]
            
            # --- Fast KNN using numpy ---
            # Get distance vector as numpy array for speed
            distances_series = distance_matrix.loc[row_name]
            
            # Remove self-match
            distances_series = distances_series[distances_series.index != row_name]
            
            # Convert to numpy for fast sorting
            distances_array = distances_series.values
            indices_array = distances_series.index.values
            
            # Get top-k indices using argpartition (faster than full sort for large k)
            k = int(n or 5)
            if k >= len(distances_array):
                # If k >= array size, just sort everything
                sorted_indices = np.argsort(distances_array)
            else:
                # Partial sort: O(n) instead of O(n log n)
                partition_indices = np.argpartition(distances_array, k)[:k]
                sorted_indices = partition_indices[np.argsort(distances_array[partition_indices])]
            
            # Get top-k closest shapes
            top_k_indices = indices_array[sorted_indices]
            top_k_distances = distances_array[sorted_indices]

            # --- Load dataset and map to unified filenames ---
            df_candidates = None
            if dataset:
                try:
                    df_candidates = get_cached_dataset_data(dataset)
                except Exception:
                    df_candidates = None
            if df_candidates is None or df_candidates.empty:
                df_candidates = file_df

            if df_candidates is None or df_candidates.empty:
                return []

            # --- Map closest filenames to unified format and dataset rows ---
            results = []
            for idx, (name, dist) in enumerate(zip(top_k_indices, top_k_distances)):
                m2 = re.match(r"([A-Za-z0-9]+)_", name)
                if not m2:
                    continue
                other_id = m2.group(1)
                unified_name = f"{other_id}_unified.obj"

                row = df_candidates[df_candidates['filename'] == unified_name]
                if not row.empty:
                    rec = row.iloc[0].to_dict()
                    rec['distance'] = float(dist)
                    results.append(rec)
                
                # Stop once we have k results
                if len(results) >= k:
                    break

            return results

        except Exception as e:
            print(f"❌ Error retrieving closest shapes: {e}")
            import traceback
            traceback.print_exc()
            return []


    # 5) Similar shapes rendering
    @app.callback(
        [Output('aux-plots-content', 'children'),
         Output('similar-shapes-accuracy', 'children'),
         Output('similar-shapes-accuracy', 'style')],
        [Input('find-shapes-button', 'n_clicks'),
        Input('amount-plots-slider', 'value'),
        Input('selected-file-store', 'data'),
        Input('aux-display-options', 'value'),
        Input('color-selector', 'value')],
        [State('selected-dataset-store', 'data')],
        prevent_initial_call=True
    )
    def render_or_clear_aux_plots(n_clicks, n_plots, selected_idx, aux_display_opts, mesh_color, selected_dataset):
        """
        Render auxiliary plots of similar shapes when the button is clicked.

        Parameters:
        - n_clicks: int, number of times the "Find Similar Shapes" button was clicked
        - n_plots: int, number of similar shapes to display
        - selected_idx: dict, selected file data with 'filename' and 'dataset' keys
        - aux_display_opts: list of str, display options for aux plots (e.g., 'wireframe', 'smooth_shading')
        - mesh_color: str, color selected for the mesh
        - selected_dataset: str, name of the selected dataset
        Returns:
        - Tuple: (list of plot divs, accuracy text, accuracy style dict)
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Clear the row when a new shape is selected
        if triggered_id == 'selected-file-store':
            return [], [], {'display': 'none'}

        # Check if we should render (button click OR display options changed while we have shapes)
        # If display options or color changed, we need to have a valid selection and prior button click
        if triggered_id in ['aux-display-options', 'color-selector', 'amount-plots-slider']:
            # Only re-render if we have a selected shape and button was clicked at least once
            if not selected_idx or not n_clicks or n_clicks <= 0:
                return no_update, no_update, no_update
            # Continue to render with new display options
        elif triggered_id == 'find-shapes-button':
            # Button click - proceed normally
            if not n_clicks or n_clicks <= 0 or selected_idx is None:
                return no_update, no_update, no_update
        else:
            # Unknown trigger, don't update
            return no_update, no_update, no_update

        # Get current shape's category
        current_category = None
        try:
            if isinstance(selected_idx, dict):
                filename = selected_idx.get('filename')
                dataset = selected_idx.get('dataset', selected_dataset)
                if filename and dataset:
                    file_df = get_cached_dataset_data(dataset)
                    matching_rows = file_df[file_df['filename'] == filename]
                    if not matching_rows.empty:
                        current_category = matching_rows.iloc[0]['category']
        except Exception as e:
            print(f"Error getting current shape category: {e}")

        show_wire = 'wireframe' in (aux_display_opts or [])
        smooth_shading = 'smooth_shading' in (aux_display_opts or [])
        total = int(n_plots or 5)

        samples = retrieve_closest_shapes(selected_idx, total)
        if not samples:
            # Return an empty list (the UI will hide the loading message when this content is set)
            return [], [], {'display': 'none'}
        
        # Calculate accuracy if we have current category
        accuracy_text = []
        accuracy_style = {'display': 'none'}
        if current_category:
            same_category_count = sum(1 for sample in samples if sample.get('category') == current_category)
            accuracy_percentage = (same_category_count / len(samples)) * 100
            accuracy_text = f"Accuracy: {accuracy_percentage:.1f}%"
            accuracy_style = {
                'fontSize': '14px',
                'fontWeight': 'bold',
                'color': '#2563eb',
                'padding': '4px 12px',
                'backgroundColor': 'rgba(37, 99, 235, 0.1)',
                'borderRadius': '6px',
                'display': 'block'
            }
        
        # Use shared category color map for consistency
        category_color_map = CATEGORY_COLOR_MAP

        cards = []
        for i, sample_row in enumerate(samples[:total]):

            # Attempt to load actual mesh file for the sampled row
            verts = None
            faces = None
            filename_str = sample_row.get('filename') or sample_row.get('file', None) or f"similar_{i+1}"
            title = filename_str
            # remove file suffix if present
            if title.lower().endswith('_unified.obj'):
                title = title.replace('_unified.obj', '')
            elif title.lower().endswith('_fill_holes_and_orientation_06.obj'):
                title = title.replace('_06_fill_holes_and_orientation.obj', '')
            elif title.lower().endswith('.obj'):
                title = title.replace('.obj', '')
                
            category_name = sample_row.get('category') or 'Unknown'
            distance = sample_row.get('distance', -1)
            distance = sample_row.get('distance', -1)
            similarity_score = sample_row.get('distance', -1)
            try:
                # get_step_file_path expects a pandas Series
                import pandas as _pd
                row_series = _pd.Series(sample_row)
                file_path, actual_step, step_info = get_step_file_path(row_series, 5)
                if file_path:
                    from pathlib import Path as _Path
                    p = _Path(file_path)
                    if p.exists():
                        verts, faces = OBJParser.parse_obj_file(str(p))
            except Exception:
                verts = None
                faces = None

            # Fallback to small random mesh if loading failed
            if verts is None or faces is None or len(verts) == 0:
                verts = (np.random.rand(100, 3) * 2 - 1)
                faces = np.array([[0, 1, 2], [1, 2, 3]])

            # Choose color from category map, fallback to provided mesh_color or default
            assigned_color = category_color_map.get(category_name, mesh_color or 'lightblue')

            fig = create_3d_plot(np.copy(verts), np.copy(faces), title, show_wireframe=show_wire,
                                mesh_color=assigned_color,
                                smooth_shading=smooth_shading,
                                camera_config=None,
                                use_rotated_vertices=False)
            # Header: show filename and a small color badge + category
            badge = html.Span(style={
                'display': 'inline-block',
                'width': '12px',
                'height': '12px',
                'backgroundColor': assigned_color,
                'borderRadius': '6px',
                'marginRight': '8px',
                'border': '1px solid rgba(0,0,0,0.12)'
            })

                        # Determine dataset to include in the pattern-matching id for the info button
            aux_dataset = sample_row.get('dataset') if isinstance(sample_row, dict) else None
            if not aux_dataset and isinstance(selected_idx, dict):
                aux_dataset = selected_idx.get('dataset')
            if not aux_dataset:
                aux_dataset = default_dataset

            # Inline small info button (placed inside the header row)
            info_btn_inline = html.Button('ℹ️', id={'type': 'show-aux-descriptors', 'filename': filename_str, 'dataset': aux_dataset},
                                          title='Show shape info', n_clicks=0,
                                          style={
                                              'width': '22px', 'height': '22px', 'borderRadius': '11px',
                                              'backgroundColor': 'rgba(255,255,255,0.98)', 'border': '1px solid #ddd',
                                              'boxShadow': '0 1px 2px rgba(0,0,0,0.06)', 'marginRight': '6px',
                                              'fontSize': '12px', 'lineHeight': '16px', 'padding': '0'
                                          })

            # Tighter badge for color
            small_badge = html.Span(style={
                'display': 'inline-block', 'width': '10px', 'height': '10px', 'backgroundColor': assigned_color,
                'borderRadius': '6px', 'marginRight': '6px', 'border': '1px solid rgba(0,0,0,0.12)'
            })

            # Title span with ellipsis to avoid overflow
            title_span = html.Span([small_badge, html.Strong(title)], style={
                'display': 'inline-block', 'maxWidth': '320px', 'min-width': 'fit-content', 'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'verticalAlign': 'middle', 'marginRight': '6px'
            })

            # Category small text
            category_span = html.Span(category_name, style={'fontSize': '1em', 'font-weight': 'bold', 'color': '#666', 'marginLeft': '4px', 'flex': '0 0 auto'})

            # Build header row with left (info+title+category) and right (similarity) areas
            left_header = html.Div([
                info_btn_inline,
                title_span,
                category_span
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'flex': '1 1 auto', 'minWidth': '0'})

            similarity_div = html.Div(f"Similarity Score: {similarity_score:.3f}", style={
                'backgroundColor': 'rgba(255,255,255,0.95)', 'padding': '4px 8px',
                'borderRadius': '12px', 'fontSize': '0.82em', 'boxShadow': '0 1px 3px rgba(0,0,0,0.12)', 'flex': '0 0 110px', 'textAlign': 'right'
            })

            header_row = html.Div([left_header, similarity_div], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'gap': '8px'})

            card = html.Div([
                header_row,
                dcc.Graph(figure=fig, className='three-d-plot')
            ], style={
                'minWidth': '360px',
                'height': '200px',
                'backgroundColor': '#fff',
                'border': '1px solid #e1e1e1',
                'borderRadius': '8px',
                'boxShadow': '0 1px 4px rgba(0,0,0,0.06)',
                'padding': '6px',
                'position': 'relative',
                'overflow': 'hidden'
            })
            cards.append(card)

        # Return cards, accuracy text, and accuracy style
        return cards, accuracy_text, accuracy_style

    # 6) Clustering modal (t-SNE visualization)
    @app.callback(
        Output('clustering-modal-open', 'data', allow_duplicate=True),
        [Input('show-clustering-btn', 'n_clicks'), Input('clustering-modal-hidden-close-trigger', 'n_clicks')],
        prevent_initial_call=True
    )
    def set_clustering_modal_open(show_clicks, hidden_close_clicks):
        """Set the clustering-modal-open store based on which button triggered."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered == 'show-clustering-btn':
            return True
        if triggered == 'clustering-modal-hidden-close-trigger':
            return False
        return no_update

    @app.callback(
        [Output('clustering-modal', 'children'), Output('clustering-modal', 'style')],
        Input('clustering-modal-open', 'data'),
        [State('selected-file-store', 'data'), State('amount-plots-slider', 'value')],
        prevent_initial_call=False
    )
    def show_clustering_modal(is_open, selected_file_data, n_neighbors):
        """Build and display the clustering modal with interactive t-SNE plot."""
        if not is_open:
            return [], {'display': 'none'}
        
        # Default background opacity
        bg_opacity = 0.1
        
        # Helper function to clean shape names
        def clean_shape_name(filename):
            """Remove processing step suffixes from filenames for display."""
            if not filename:
                return filename
            # Remove common suffixes
            name = filename
            suffixes_to_remove = [
                '_06_fill_holes_and_orientation.obj',
                '_fill_holes_and_orientation_06.obj',
                '_unified.obj',
                '.obj'
            ]
            for suffix in suffixes_to_remove:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            return name
        
        # Load t-SNE data
        embedding_df, labels_df = get_cached_tsne_data()
        
        if embedding_df is None or labels_df is None:
            error_content = html.Div([
                html.Div([
                    html.H3("❌ t-SNE Data Not Available", style={'color': '#e74c3c', 'marginBottom': '15px'}),
                    html.P("The t-SNE embedding files were not found. Please ensure the following files exist:"),
                    html.Ul([
                        html.Li("Src/scalability/topology_graph.csv"),
                        html.Li("Src/scalability/class_labels.csv")
                    ]),
                    html.P("You may need to run the topology graph generation script first."),
                    html.Button('Close', n_clicks=0,
                               className='modal-close-btn',
                               style={'marginTop': '20px', 'padding': '8px 20px',
                                     'backgroundColor': '#e74c3c', 'color': 'white',
                                     'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'})
                ], className='modal-content', style={'padding': '30px', 'maxWidth': '600px'})
            ], className='modal-backdrop', style={'display': 'flex'})
            
            return error_content, {'display': 'flex'}
        
        # Merge embedding with labels
        merged = embedding_df.merge(labels_df, left_index=True, right_on="shape", how="left")
        
        # Encode classes for consistent coloring
        merged["class"] = merged["class"].astype(str)
        ordered_classes = sorted(merged["class"].dropna().unique())
        
        # Use shared category color map for consistency across the app
        color_map = CATEGORY_COLOR_MAP
        
        # Determine focus mode if shape selected
        target_shape = None
        target_class = None
        neighbors_list = []
        
        if selected_file_data and isinstance(selected_file_data, dict):
            filename = selected_file_data.get('filename')
            if filename:
                # Try exact match first
                if filename in merged['shape'].values:
                    target_shape = filename
                else:
                    # Try matching by ID prefix (e.g., m1338 from m1338_unified.obj or m1338_06_*.obj)
                    import re
                    m = re.match(r"([A-Za-z0-9]+)_", filename)
                    if m:
                        shape_id = m.group(1)
                        # Find any shape with this ID prefix
                        matching_shapes = [s for s in merged['shape'].values if s.startswith(shape_id + "_")]
                        if matching_shapes:
                            target_shape = matching_shapes[0]  # Use first match (typically _06 version)
                            print(f"🎯 Matched {filename} to {target_shape} for t-SNE focus mode")
                
                if target_shape:
                    target_row = merged.loc[merged['shape'] == target_shape].iloc[0]
                    target_class = target_row['class']
                    
                    # Get neighbors from distance matrix
                    dist_matrix = get_cached_distance_matrix()
                    if dist_matrix is not None and target_shape in dist_matrix.index:
                        row = dist_matrix.loc[target_shape].copy().drop(labels=[target_shape], errors="ignore")
                        row = pd.to_numeric(row, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
                        n_neigh = max(0, int(n_neighbors or 5))
                        neighbors_list = row.nsmallest(n_neigh).index.tolist() if n_neigh > 0 else []
                        print(f"✅ Found {len(neighbors_list)} neighbors for {target_shape}")
        
        # Create plot data with different opacity/marker settings
        if target_shape:
            # Focus mode: match topology_graph.py behavior
            # - Background (all points): 10% opacity
            # - Same class (not neighbors): 80% opacity
            # - Neighbors (both same and diff class): triangles with black edge
            # - Target: square with black edge
            
            base_size = 15  # Uniform size like topology_graph.py
            
            # All background points (10% opacity)
            background_mask = ~merged['shape'].isin([target_shape] + neighbors_list) & (merged['class'] != target_class)
            
            # Same class (not target, not neighbors) - 100% opacity
            same_class_mask = (merged['class'] == target_class) & (merged['shape'] != target_shape) & ~merged['shape'].isin(neighbors_list)
            
            # Neighbors (both same and different class)
            neighbors_same_class_mask = merged['shape'].isin(neighbors_list) & (merged['class'] == target_class)
            neighbors_diff_class_mask = merged['shape'].isin(neighbors_list) & (merged['class'] != target_class)
            
            # Target
            target_mask = merged['shape'] == target_shape
            
            fig = go.Figure()
            
            # Add background points with configurable opacity
            if background_mask.any():
                for cls in merged.loc[background_mask, 'class'].unique():
                    if cls not in color_map:
                        continue
                    cls_mask = background_mask & (merged['class'] == cls)
                    # Clean shape names for hover tooltips
                    hover_texts = [clean_shape_name(s) for s in merged.loc[cls_mask, 'shape']]
                    fig.add_trace(go.Scatter(
                        x=merged.loc[cls_mask, 'x'],
                        y=merged.loc[cls_mask, 'y'],
                        mode='markers',
                        name=cls,
                        text=hover_texts,
                        hovertemplate='<b>%{text}</b><br>Class: ' + cls + '<extra></extra>',
                        marker=dict(size=base_size, color=color_map[cls], opacity=bg_opacity),
                        showlegend=True,
                        legendgroup=cls
                    ))
            
            # Add same class points (80% opacity)
            if same_class_mask.any():
                hover_texts = [clean_shape_name(s) for s in merged.loc[same_class_mask, 'shape']]
                fig.add_trace(go.Scatter(
                    x=merged.loc[same_class_mask, 'x'],
                    y=merged.loc[same_class_mask, 'y'],
                    mode='markers',
                    name=f"{target_class} (same class)",
                    text=hover_texts,
                    hovertemplate='<b>%{text}</b><br>Class: ' + target_class + '<extra></extra>',
                    marker=dict(size=base_size, color=color_map.get(target_class, '#999999'), opacity=0.8),
                    showlegend=False,
                    legendgroup=target_class
                ))
            # Add same-class neighbors (triangles, 100% opacity, black edge)
            if neighbors_same_class_mask.any():
                hover_texts = [clean_shape_name(s) for s in merged.loc[neighbors_same_class_mask, 'shape']]
                fig.add_trace(go.Scatter(
                    x=merged.loc[neighbors_same_class_mask, 'x'],
                    y=merged.loc[neighbors_same_class_mask, 'y'],
                    mode='markers',
                    name=f"Same-class neighbors",
                    text=hover_texts,
                    hovertemplate='<b>%{text}</b><br>Class: ' + target_class + ' (neighbor)<extra></extra>',
                    marker=dict(size=base_size, color=color_map.get(target_class, '#999999'), opacity=1.0, 
                               symbol='triangle-up', line=dict(width=0.4, color='black')),
                    showlegend=True
                ))
            
            # Add different-class neighbors (triangles, 100% opacity, black edge)
            if neighbors_diff_class_mask.any():
                for cls in merged.loc[neighbors_diff_class_mask, 'class'].unique():
                    if cls not in color_map:
                        continue
                    cls_neigh_mask = neighbors_diff_class_mask & (merged['class'] == cls)
                    hover_texts = [clean_shape_name(s) for s in merged.loc[cls_neigh_mask, 'shape']]
                    fig.add_trace(go.Scatter(
                        x=merged.loc[cls_neigh_mask, 'x'],
                        y=merged.loc[cls_neigh_mask, 'y'],
                        mode='markers',
                        name=f"{cls} (neighbor)",
                        text=hover_texts,
                        hovertemplate='<b>%{text}</b><br>Class: ' + cls + ' (neighbor)<extra></extra>',
                        marker=dict(size=base_size, color=color_map[cls], opacity=1.0,
                                   symbol='triangle-up', line=dict(width=0.4, color='black')),
                        showlegend=True,
                        legendgroup=cls
                    ))
            
            # Add target point (square, 100% opacity, black edge)
            if target_mask.any():
                hover_texts = [clean_shape_name(s) for s in merged.loc[target_mask, 'shape']]
                fig.add_trace(go.Scatter(
                    x=merged.loc[target_mask, 'x'],
                    y=merged.loc[target_mask, 'y'],
                    mode='markers',
                    name=f"{clean_shape_name(target_shape)} (target)",
                    text=hover_texts,
                    hovertemplate='<b>%{text}</b><br>Class: ' + target_class + ' (TARGET)<extra></extra>',
                    marker=dict(size=base_size, color=color_map.get(target_class, '#999999'), opacity=1.0,
                               symbol='square', line=dict(width=0.6, color='black')),
                    showlegend=True
                ))
            
            title_text = f"t-SNE Embedding • Focus: {clean_shape_name(target_shape)} (n={len(neighbors_list)})"
            # title_text = f"t-SNE Embedding • Focus: {target_shape} (n={len(neighbors_list)})"
        else:
            # Normal mode: show all points by class with consistent colors
            base_size = 15
            fig = go.Figure()
            
            for cls in ordered_classes:
                if cls not in color_map:
                    continue
                cls_mask = merged['class'] == cls
                hover_texts = [clean_shape_name(s) for s in merged.loc[cls_mask, 'shape']]
                fig.add_trace(go.Scatter(
                    x=merged.loc[cls_mask, 'x'],
                    y=merged.loc[cls_mask, 'y'],
                    mode='markers',
                    name=cls,
                    text=hover_texts,
                    hovertemplate='<b>%{text}</b><br>Class: ' + cls + '<extra></extra>',
                    marker=dict(size=base_size, color=color_map[cls], opacity=0.8),
                    showlegend=True
                ))
            
            title_text = "t-SNE Embedding of 3D Shapes"
        
        fig.update_layout(
            title=title_text,
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            hovermode='closest',
            width=1400,
            height=800,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            margin=dict(l=50, r=250, t=80, b=50)
        )
        
        # Build modal content
        modal_content = html.Div([
            html.Div([
                html.Div([
                    html.H3("🧬 t-SNE Clustering Visualization", style={'margin': '0 0 20px 0', 'color': '#2c3e50'}),
                    html.Button('✕', n_clicks=0,
                               className='modal-close-x',
                               style={
                                   'position': 'absolute', 'top': '15px', 'right': '15px',
                                   'background': 'none', 'border': 'none', 'fontSize': '24px',
                                   'cursor': 'pointer', 'color': '#7f8c8d', 'lineHeight': '1'
                               })
                ], style={'position': 'relative', 'borderBottom': '2px solid #ecf0f1', 'paddingBottom': '15px', 'marginBottom': '20px'}),
                
                # Controls row: Opacity and Family Filter
                html.Div([
                    # Opacity control (only in focus mode)
                    html.Div([
                        html.Label("Background Opacity:", style={'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '14px'}),
                        dcc.RadioItems(
                            id='tsne-background-opacity-toggle',
                            options=[
                                {'label': ' Low (0.1)', 'value': 0.1},
                                {'label': ' Medium (0.35)', 'value': 0.35},
                                {'label': ' Full (1.0)', 'value': 1.0}
                            ],
                            value=0.1,
                            inline=True,
                            style={'fontSize': '14px'}
                        )
                    ], style={
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '6px',
                        'display': 'inline-flex',
                        'alignItems': 'center',
                        'marginRight': '15px'
                    }) if target_shape else None,
                    
                    # Family filter with select all/clear all buttons
                    html.Div([
                        html.Label("Filter Families:", style={'marginRight': '10px', 'fontWeight': 'bold', 'fontSize': '14px', 'whiteSpace': 'nowrap'}),
                        html.Div([
                            dcc.Dropdown(
                                id='tsne-family-filter',
                                options=[
                                    {'label': '🛩️ Aircraft', 'value': 'aircraft'},
                                    {'label': '🚗 Ground Vehicles', 'value': 'ground_vehicles'},
                                    {'label': '🚢 Water Vessels', 'value': 'water'},
                                    {'label': '🏢 Buildings', 'value': 'buildings'},
                                    {'label': '🪑 Furniture', 'value': 'furniture'},
                                    {'label': '🎵 Music', 'value': 'music'},
                                    {'label': '💻 Electronics', 'value': 'electronics'},
                                    {'label': '💡 Lighting', 'value': 'lighting'},
                                    {'label': '🍶 Small Objects', 'value': 'small_objects'},
                                    {'label': '🔫 Weapons & Tools', 'value': 'weapons_tools'},
                                    {'label': '🌳 Nature', 'value': 'nature'},
                                    {'label': '👤 Living', 'value': 'living'},
                                    {'label': '♟️ Misc', 'value': 'misc'}
                                ],
                                value=['aircraft', 'ground_vehicles', 'water', 'buildings', 'furniture', 'music', 
                                       'electronics', 'lighting', 'small_objects', 'weapons_tools', 'nature', 'living', 'misc'],
                                multi=True,
                                placeholder="Select families to display...",
                                style={'fontSize': '13px', 'width': '400px'}
                            ),
                            html.Div([
                                html.Button('Select All', id='tsne-select-all-families', n_clicks=0,
                                           style={
                                               'padding': '4px 12px',
                                               'fontSize': '12px',
                                               'backgroundColor': '#3498db',
                                               'color': 'white',
                                               'border': 'none',
                                               'borderRadius': '4px',
                                               'cursor': 'pointer',
                                               'marginRight': '5px'
                                           }),
                                html.Button('Clear All', id='tsne-clear-all-families', n_clicks=0,
                                           style={
                                               'padding': '4px 12px',
                                               'fontSize': '12px',
                                               'backgroundColor': '#95a5a6',
                                               'color': 'white',
                                               'border': 'none',
                                               'borderRadius': '4px',
                                               'cursor': 'pointer'
                                           })
                            ], style={'marginLeft': '10px', 'display': 'flex', 'alignItems': 'center'})
                        ], style={'display': 'flex', 'alignItems': 'center'})
                    ], style={
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '6px',
                        'display': 'inline-flex',
                        'alignItems': 'center',
                        'flex': '1'
                    })
                ], style={
                    'marginBottom': '15px',
                    'display': 'flex',
                    'alignItems': 'center',
                    'width': '100%'
                }),
                
                dcc.Graph(id='tsne-plot-graph', figure=fig, config={'displayModeBar': True, 'displaylogo': False}, style={'width': '100%'}),
                
                html.Div([
                    html.P([
                        html.Strong("💡 Tip: "),
                        "Hover over points to see shape names. ",
                        "Select a shape from the left panel and click this button again to focus on it and its neighbors!"
                    ], style={'color': '#7f8c8d', 'fontSize': '14px', 'marginTop': '15px'})
                ])
            ], className='modal-content', style={
                'backgroundColor': 'white',
                'padding': '30px',
                'borderRadius': '12px',
                'boxShadow': '0 10px 40px rgba(0,0,0,0.2)',
                'width': '1500px',
                'maxWidth': '95vw',
                'maxHeight': '95vh',
                'overflow': 'auto',
                'position': 'relative'
            })
        ], className='modal-backdrop', style={
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(0, 0, 0, 0.7)',
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'zIndex': 2000
        })
        
        return modal_content, {'display': 'flex'}

    @app.callback(
        Output('tsne-plot-graph', 'figure'),
        Input('tsne-background-opacity-toggle', 'value'),
        [State('tsne-plot-graph', 'figure'), State('selected-file-store', 'data')],
        prevent_initial_call=True
    )
    def update_tsne_opacity(new_opacity, current_figure, selected_file_data):
        """Update background point opacity in the t-SNE plot."""
        if current_figure is None or new_opacity is None:
            return no_update
        
        # Get selected shape if any
        selected_shape = None
        if selected_file_data and 'filename' in selected_file_data:
            selected_shape = selected_file_data['filename']
        elif selected_file_data and 'file_path' in selected_file_data:
            selected_shape = os.path.basename(selected_file_data['file_path'])
        
        # Only update if in focus mode (selected_shape exists)
        if not selected_shape:
            return no_update
        
        # Create a new figure with updated opacities
        import plotly.graph_objects as go
        fig = go.Figure(current_figure)
        
        # Update background traces (those without special markers or keywords)
        for i, trace in enumerate(fig.data):
            trace_name = trace.name if trace.name else ''
            marker_symbol = trace.marker.symbol if hasattr(trace, 'marker') and hasattr(trace.marker, 'symbol') else None
            
            # Background traces are those that don't have special markers
            # and don't have "neighbor", "target", or "same class" in the name
            is_background = (
                'neighbor' not in trace_name.lower() and 
                'target' not in trace_name.lower() and 
                'same class' not in trace_name.lower() and
                marker_symbol not in ['triangle-up', 'square']
            )
            
            if is_background and hasattr(trace, 'marker'):
                # Update opacity for background trace
                fig.data[i].marker.opacity = new_opacity
        
        return fig

    @app.callback(
        Output('tsne-plot-graph', 'figure', allow_duplicate=True),
        Input('tsne-family-filter', 'value'),
        [State('tsne-plot-graph', 'figure')],
        prevent_initial_call=True
    )
    def update_tsne_family_filter(selected_families, current_figure):
        """Filter t-SNE plot traces based on selected family groups."""
        if current_figure is None or selected_families is None:
            return no_update
        
        # Import category groups
        from .category_colors import CATEGORY_GROUPS
        
        # Build set of categories to show based on selected families
        categories_to_show = set()
        for family in selected_families:
            if family in CATEGORY_GROUPS:
                categories_to_show.update(CATEGORY_GROUPS[family])
        
        # Create a new figure with filtered traces
        import plotly.graph_objects as go
        fig = go.Figure(current_figure)
        
        # Update visibility for each trace based on whether its category is in the selected families
        for i, trace in enumerate(fig.data):
            trace_name = trace.name if trace.name else ''
            
            # Special traces (neighbors, target) are always visible
            is_special = (
                'neighbor' in trace_name.lower() or 
                'target' in trace_name.lower() or
                'same class' in trace_name.lower()
            )
            
            if is_special:
                # Always show special traces
                fig.data[i].visible = True
            else:
                # For regular category traces, check if category is in selected families
                # The trace name is the category name (unless it has special suffixes)
                category_name = trace_name.split(' (')[0].strip()  # Remove any suffixes like " (neighbor)"
                
                # Show trace if its category is in the allowed set
                fig.data[i].visible = category_name in categories_to_show
        
        return fig

    @app.callback(
        Output('tsne-family-filter', 'value'),
        [Input('tsne-select-all-families', 'n_clicks'),
         Input('tsne-clear-all-families', 'n_clicks')],
        prevent_initial_call=True
    )
    def update_family_selection(select_all_clicks, clear_all_clicks):
        """Handle Select All and Clear All buttons for family filter."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'tsne-select-all-families':
            # Select all families
            return ['aircraft', 'ground_vehicles', 'water', 'buildings', 'furniture', 'music', 
                    'electronics', 'lighting', 'small_objects', 'weapons_tools', 'nature', 'living', 'misc']
        elif trigger_id == 'tsne-clear-all-families':
            # Clear all families
            return []
        
        return no_update

    # Control the modal visibility via a persistent Store to avoid missing-id issues
    @app.callback(
        Output('global-descriptors-open', 'data', allow_duplicate=True),
        [Input('show-global-descriptors-btn', 'n_clicks'), Input('global-descriptors-hidden-close-trigger', 'n_clicks')],
        prevent_initial_call=True
    )
    def set_global_descriptors_open(show_clicks, hidden_close_clicks):
        """Set the `global-descriptors-open` store based on which button triggered.

        We listen to the persistent hidden close trigger (`global-descriptors-hidden-close-trigger`)
        instead of any in-modal id to avoid missing-id validation errors in the renderer.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered == 'show-global-descriptors-btn':
            return True
        if triggered == 'global-descriptors-hidden-close-trigger':
            return False
        return no_update

    # Main modal builder: listens to the store and renders/hides the modal atomically
    @app.callback(
        [Output('global-descriptors-modal', 'children'), Output('global-descriptors-modal', 'style')],
        Input('global-descriptors-open', 'data'),
        [State('selected-file-store', 'data'), State('selected-dataset-store', 'data')],
        prevent_initial_call=False
    )
    def show_global_descriptors(is_open, selected_file_data, selected_dataset):
        """Display or hide the modal depending on the store value.

        When is_open is True, build the modal content. When False or missing,
        hide the modal. This avoids relying on combined n_clicks Inputs.
        """
        # If store indicates closed or missing, hide modal
        if not is_open:
            return [], {'display': 'none'}

        # Modal base style (centered overlay)
        modal_style = {
            'display': 'block',
            'position': 'fixed',
            'left': '0',
            'top': '0',
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(0,0,0,0.5)',
            'zIndex': 9999,
            'padding': '40px',
            'boxSizing': 'border-box',
            'overflow': 'auto'
        }

        # Simple header used for early-return error messages (no shape selected / dataset errors)
        header = html.Div([
            html.Div(html.H3("Global Descriptor Histograms", style={'margin': 0, 'color': '#000'}), style={'flex': '1'}),
            # In-modal Close button: plain Dash button (clicks are proxied by assets JS)
            html.Button('Close', n_clicks=0,
                        style={'background': '#fff', 'border': 'none', 'padding': '6px 10px', 'borderRadius': '6px', 'cursor': 'pointer'})
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '12px'})

        # If no shape selected, show helpful message
        if not selected_file_data or not isinstance(selected_file_data, dict):
            body = html.Div([
                html.P("No shape selected. Please select a shape from the list first.", style={'color': '#000'})
            ])
            content_card = html.Div([header, body], style={'maxWidth': '1100px', 'margin': '0 auto', 'background': '#fff', 'color': '#000', 'padding': '16px', 'borderRadius': '8px', 'boxShadow': '0 8px 30px rgba(0,0,0,0.25)'})
            # Wrap content in the full-screen modal overlay style so it appears as a dialog
            return html.Div(content_card, style=modal_style), modal_style

        selected_filename = selected_file_data.get('filename')
        dataset = selected_file_data.get('dataset', selected_dataset)
        if not dataset:
            dataset = 'Data'

        # Load file data
        try:
            df = get_cached_dataset_data(dataset)
        except Exception as e:
            body = html.Div([html.P(f"Failed to load dataset: {e}", style={'color': '#000'})])
            content_card = html.Div([header, body], style={'maxWidth': '1100px', 'margin': '0 auto', 'background': '#fff', 'color': '#000', 'padding': '16px', 'borderRadius': '8px', 'boxShadow': '0 8px 30px rgba(0,0,0,0.25)'})
            return html.Div(content_card, style=modal_style), modal_style

        if df is None or df.empty:
            body = html.Div([html.P("Dataset is empty or unavailable.", style={'color': '#000'})])
            content_card = html.Div([header, body], style={'maxWidth': '1100px', 'margin': '0 auto', 'background': '#fff', 'color': '#000', 'padding': '16px', 'borderRadius': '8px', 'boxShadow': '0 8px 30px rgba(0,0,0,0.25)'})
            return html.Div(content_card, style=modal_style), modal_style

        matching = df[df['filename'] == selected_filename]
        if matching.empty:
            body = html.Div([html.P(f"Selected file '{selected_filename}' not found in dataset.", style={'color': '#000'})])
            content_card = html.Div([header, body], style={'maxWidth': '1100px', 'margin': '0 auto', 'background': '#fff', 'color': '#000', 'padding': '16px', 'borderRadius': '8px', 'boxShadow': '0 8px 30px rgba(0,0,0,0.25)'})
            return html.Div(content_card, style=modal_style), modal_style

        row = matching.iloc[0]

        # Determine display name for the selected shape (try several possible fields)
        display_name = None
        for name_key in ('name', 'Name', 'analysis_name', 'display_name', 'shape_file'):
            if name_key in row.index and pd.notna(row.get(name_key)) and str(row.get(name_key)).strip() != '':
                display_name = str(row.get(name_key))
                break
        if not display_name:
            display_name = selected_filename or 'Unknown'

        # Update header to include the shape name
        header = html.Div([
            html.Div(html.H3(f"Global Descriptor Histograms — {display_name}", style={'margin': 0, 'color': '#000'}), style={'flex': '1'}),
            # In-modal Close button: plain Dash button (clicks are proxied by assets JS)
            html.Button('Close', n_clicks=0,
                        style={'background': '#fff', 'border': 'none', 'padding': '6px 10px', 'borderRadius': '6px', 'cursor': 'pointer'})
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '12px'})

        # Descriptor pairs to render
        descriptor_pairs = [
            ('A3', 'A3_hist', 'A3_bins'),
            ('D1', 'D1_hist', 'D1_bins'),
            ('D2', 'D2_hist', 'D2_bins'),
            ('D3', 'D3_hist', 'D3_bins'),
            ('D4', 'D4_hist', 'D4_bins')
        ]
        # Color palette for the five histograms (A3, D1, D2, D3, D4)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        graphs = []
        inline_graphs = []  # smaller versions for inline display
        for title, hist_field, bins_field in descriptor_pairs:
            # Accept several possible column name variants
            hist_val = None
            bins_val = None
            for key in (hist_field, hist_field.lower(), 'analysis_' + hist_field, 'analysis_' + hist_field.lower()):
                if key in row.index:
                    hist_val = row.get(key)
                    break
            for key in (bins_field, bins_field.lower(), 'analysis_' + bins_field, 'analysis_' + bins_field.lower()):
                if key in row.index:
                    bins_val = row.get(key)
                    break

            # Try robust parsing of stored histogram values
            mids, hist_vals = _parse_hist_and_bins(hist_val, bins_val)
            # If parser failed, try a simple regex-based numeric extraction as a last resort
            if (mids is None or hist_vals is None) and isinstance(hist_val, str):
                try:
                    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", hist_val)]
                    if nums:
                        hist_vals = nums
                        mids = list(range(len(nums)))
                        print(f"[debug] Fallback parsed {len(nums)} histogram values for {title} from string for file {selected_filename}")
                except Exception as e:
                    print(f"[debug] Fallback parsing failed for {title}: {e}")
            # Debug: log when histogram parsing fails for visibility in server logs
            if mids is None or hist_vals is None:
                print(f"[debug] _parse_hist_and_bins returned None for {title} on file {selected_filename}. hist_val repr: {repr(hist_val)} bins_val repr: {repr(bins_val)}")
            if mids is None or hist_vals is None:
                # Placeholder card for modal — give it an opaque white background and dark text
                card = html.Div([
                    html.H4(f"{title} - Data unavailable", style={'color': '#000'}),
                    html.P("Histogram or bins data missing for this shape.", style={'color': '#333'})
                ], style={'flex': '1', 'minWidth': '200px', 'padding': '12px', 'background': '#fff', 'borderRadius': '6px'})
                # Inline placeholder (smaller)
                inline_card = html.Div([
                    html.H5(f"{title}", style={'margin': '6px 0', 'color': '#000'}),
                    html.P("Data unavailable", style={'margin': 0, 'fontSize': '0.85em', 'color': '#666'})
                ], style={'flex': '0 0 160px', 'minWidth': '140px', 'padding': '8px', 'background': '#fff', 'borderRadius': '6px', 'textAlign': 'center'})
            else:
                fig = go.Figure()
                # choose color by index
                idx = len(graphs) if len(graphs) < len(colors) else 0
                color = colors[idx]
                fig.add_trace(go.Bar(x=mids, y=hist_vals, marker_color=color))
                fig.update_layout(title=f"{title}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  xaxis_title='Value', yaxis_title='Frequency', margin=dict(l=30, r=10, t=30, b=30))

                card = html.Div([
                    dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '260px'})
                ], style={'flex': '1', 'minWidth': '200px', 'padding': '6px', 'background': '#fff', 'borderRadius': '6px'})
                # Inline smaller thumbnail version
                inline_fig = go.Figure()
                inline_fig.add_trace(go.Bar(x=mids, y=hist_vals, marker_color=color))
                inline_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                         margin=dict(l=20, r=6, t=6, b=20), xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
                inline_card = html.Div([
                    dcc.Graph(figure=inline_fig, config={'displayModeBar': False}, style={'height': '160px', 'width': '180px'})
                ], style={'flex': '0 0 180px', 'minWidth': '140px', 'padding': '6px', 'background': '#fff', 'borderRadius': '6px'})

            graphs.append(card)
            inline_graphs.append(inline_card)

        # Layout the five histograms in rows (wrap)
        graphs_row = html.Div(graphs, style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'justifyContent': 'center'})

        body = html.Div([
            graphs_row
        ])

        content_card = html.Div([header, body], style={'maxWidth': '1100px', 'margin': '0 auto', 'background': '#fff', 'color': '#000', 'padding': '16px', 'borderRadius': '8px', 'boxShadow': '0 8px 30px rgba(0,0,0,0.25)'})
        # Ensure the returned content is wrapped in the full-screen modal overlay so it appears centered
        return html.Div(content_card, style=modal_style), modal_style

    # Modal visibility is handled within the same server callback that returns children and style.

    # Aux modal: separate builder so auxiliary sample info does not overwrite main selection
    @app.callback(
        [Output('aux-descriptors-modal', 'children'), Output('aux-descriptors-modal', 'style')],
        Input('aux-descriptors-open', 'data'),
        [State('aux-selected-file-store', 'data'), State('selected-dataset-store', 'data')],
        prevent_initial_call=False
    )
    def show_aux_descriptors(is_open, aux_file_data, selected_dataset):
        """Display or hide the auxiliary modal depending on its store value.

        This mirrors `show_global_descriptors` but uses separate stores/modal ids so
        opening aux info doesn't change the main `selected-file-store`.
        """
        if not is_open:
            return [], {'display': 'none'}

        modal_style = {
            'display': 'block',
            'position': 'fixed',
            'left': '0',
            'top': '0',
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(0,0,0,0.5)',
            'zIndex': 9999,
            'padding': '40px',
            'boxSizing': 'border-box',
            'overflow': 'auto'
        }

        header = html.Div([
            html.Div(html.H3("Shape Info", style={'margin': 0, 'color': '#000'}), style={'flex': '1'}),
            html.Button('Close', n_clicks=0,
                        style={'background': '#fff', 'border': 'none', 'padding': '6px 10px', 'borderRadius': '6px', 'cursor': 'pointer'})
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '12px'})

        if not aux_file_data or not isinstance(aux_file_data, dict):
            body = html.Div([html.P("No sample selected.", style={'color': '#000'})])
            content_card = html.Div([header, body], style={'maxWidth': '900px', 'margin': '0 auto', 'background': '#fff', 'color': '#000', 'padding': '14px', 'borderRadius': '8px', 'boxShadow': '0 8px 24px rgba(0,0,0,0.22)'})
            return html.Div(content_card, style=modal_style), modal_style

        selected_filename = aux_file_data.get('filename')
        dataset = aux_file_data.get('dataset', selected_dataset)
        if not dataset:
            dataset = 'Data'

        try:
            df = get_cached_dataset_data(dataset)
        except Exception as e:
            body = html.Div([html.P(f"Failed to load dataset: {e}", style={'color': '#000'})])
            content_card = html.Div([header, body], style={'maxWidth': '900px', 'margin': '0 auto', 'background': '#fff', 'color': '#000', 'padding': '14px', 'borderRadius': '8px', 'boxShadow': '0 8px 24px rgba(0,0,0,0.22)'})
            return html.Div(content_card, style=modal_style), modal_style

        if df is None or df.empty:
            body = html.Div([html.P("Dataset is empty or unavailable.", style={'color': '#000'})])
            content_card = html.Div([header, body], style={'maxWidth': '900px', 'margin': '0 auto', 'background': '#fff', 'color': '#000', 'padding': '14px', 'borderRadius': '8px', 'boxShadow': '0 8px 24px rgba(0,0,0,0.22)'})
            return html.Div(content_card, style=modal_style), modal_style

        matching = df[df['filename'] == selected_filename]
        if matching.empty:
            body = html.Div([html.P(f"Sample '{selected_filename}' not found in dataset.", style={'color': '#000'})])
            content_card = html.Div([header, body], style={'maxWidth': '900px', 'margin': '0 auto', 'background': '#fff', 'color': '#000', 'padding': '14px', 'borderRadius': '8px', 'boxShadow': '0 8px 24px rgba(0,0,0,0.22)'})
            return html.Div(content_card, style=modal_style), modal_style

        row = matching.iloc[0]

        # Use the same Shape Info card used elsewhere so formatting matches exactly
        try:
            mesh = ShapeMesh.from_file_row(row)
            info_card = mesh.get_card_header_html()
            # Wrap mesh-provided card in an opaque white container so content is readable on the overlay
            info_card = html.Div(info_card, style={'background': 'rgb(241 237 225)', 'color': '#000', 'padding': '12px', 'margin-bottom': '1.2rem', 'borderRadius': '6px', 'boxShadow': '0 6px 20px rgba(0,0,0,0.18)'})
        except Exception as e:
            info_card = html.Div([html.H4("❌ Error building Shape Info", style={'color': '#e74c3c'}), html.Div(str(e))])

        # Render the five global descriptor histograms just like the main modal
        descriptor_pairs = [
            ('A3', 'A3_hist', 'A3_bins'),
            ('D1', 'D1_hist', 'D1_bins'),
            ('D2', 'D2_hist', 'D2_bins'),
            ('D3', 'D3_hist', 'D3_bins'),
            ('D4', 'D4_hist', 'D4_bins')
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        graphs = []
        for i, (title, hist_field, bins_field) in enumerate(descriptor_pairs):
            hist_val = None
            bins_val = None
            for key in (hist_field, hist_field.lower(), 'analysis_' + hist_field, 'analysis_' + hist_field.lower()):
                if key in row.index:
                    hist_val = row.get(key)
                    break
            for key in (bins_field, bins_field.lower(), 'analysis_' + bins_field, 'analysis_' + bins_field.lower()):
                if key in row.index:
                    bins_val = row.get(key)
                    break

            mids, hist_vals = _parse_hist_and_bins(hist_val, bins_val)
            # Fallback parsing for string-encoded histograms
            if (mids is None or hist_vals is None) and isinstance(hist_val, str):
                try:
                    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", hist_val)]
                    if nums:
                        hist_vals = nums
                        mids = list(range(len(nums)))
                        print(f"[debug] Fallback parsed {len(nums)} histogram values for {title} from string for aux file {selected_filename}")
                except Exception as e:
                    print(f"[debug] Aux fallback parsing failed for {title}: {e}")
            if mids is None or hist_vals is None:
                graphs.append(html.Div([html.H4(f"{title} - Data unavailable", style={'color': '#000'})], style={'minWidth': '180px', 'background': '#fff', 'padding': '8px', 'borderRadius': '6px'}))
            else:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=mids, y=hist_vals, marker_color=colors[i] if i < len(colors) else colors[0]))
                fig.update_layout(title=f"{title}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  xaxis_title='Value', yaxis_title='Frequency', margin=dict(l=30, r=10, t=30, b=30))
                graphs.append(html.Div(dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '220px'}), style={'minWidth': '180px'}))

        graphs_row = html.Div(graphs, style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'justifyContent': 'center'})

        body = html.Div([info_card, graphs_row])
        content_card = html.Div([header, body], style={'maxWidth': '1100px', 'margin': '0 auto', 'background': '#fff', 'color': '#000', 'padding': '16px', 'borderRadius': '8px', 'boxShadow': '0 8px 30px rgba(0,0,0,0.22)'})
        return html.Div(content_card, style=modal_style), modal_style

    # Close aux modal when hidden close trigger is fired (proxied from in-modal Close button)
    @app.callback(
        Output('aux-descriptors-open', 'data', allow_duplicate=True),
        [Input('aux-descriptors-hidden-close-trigger', 'n_clicks')],
        prevent_initial_call=True
    )
    def set_aux_descriptors_open(hidden_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update
        return False

    @app.callback(
        Output('inline-global-descriptors', 'children', allow_duplicate=True),
        [Input('selected-file-store', 'data'), Input('selected-dataset-store', 'data'), Input('shape-info', 'children')],
        prevent_initial_call='initial_duplicate'
    )
    def update_inline_descriptors(selected_file_data, selected_dataset, shape_info=None):
        """Populate the inline thumbnails below Shape Info when a shape is selected.

        This mirrors the inline part of the modal rendering but only returns the
        inline children so it can be updated independently of the modal.
        """
        try:
            if not selected_file_data or not isinstance(selected_file_data, dict):
                return []

            selected_filename = selected_file_data.get('filename')
            dataset = selected_file_data.get('dataset', selected_dataset)
            if not dataset:
                dataset = 'Data'

            df = get_cached_dataset_data(dataset)
            if df is None or df.empty:
                return []

            matching = df[df['filename'] == selected_filename]
            if matching.empty:
                return []

            row = matching.iloc[0]

            descriptor_pairs = [
                ('A3', 'A3_hist', 'A3_bins'),
                ('D1', 'D1_hist', 'D1_bins'),
                ('D2', 'D2_hist', 'D2_bins'),
                ('D3', 'D3_hist', 'D3_bins'),
                ('D4', 'D4_hist', 'D4_bins')
            ]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            inline_graphs = []
            for i, (title, hist_field, bins_field) in enumerate(descriptor_pairs):
                hist_val = None
                bins_val = None
                for key in (hist_field, hist_field.lower(), 'analysis_' + hist_field, 'analysis_' + hist_field.lower()):
                    if key in row.index:
                        hist_val = row.get(key)
                        break
                for key in (bins_field, bins_field.lower(), 'analysis_' + bins_field, 'analysis_' + bins_field.lower()):
                    if key in row.index:
                        bins_val = row.get(key)
                        break

                mids, hist_vals = _parse_hist_and_bins(hist_val, bins_val)
                # Fallback parsing for string-encoded histograms
                if (mids is None or hist_vals is None) and isinstance(hist_val, str):
                    try:
                        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", hist_val)]
                        if nums:
                            hist_vals = nums
                            mids = list(range(len(nums)))
                            print(f"[debug] Fallback parsed {len(nums)} inline histogram values for {title} from string for file {selected_filename}")
                    except Exception as e:
                        print(f"[debug] Inline fallback parsing failed for {title}: {e}")
                if mids is None or hist_vals is None:
                    inline_card = html.Div([
                        html.Div(title, style={'fontWeight': '600', 'marginBottom': '6px', 'fontSize': '12px'}),
                        html.Div("Data unavailable", style={'fontSize': '11px', 'color': '#666'})
                    ], style={'flex': '0 0 140px', 'minWidth': '120px', 'padding': '6px', 'background': '#fff', 'borderRadius': '6px', 'textAlign': 'center'})
                else:
                    color = colors[i] if i < len(colors) else colors[0]
                    inline_fig = go.Figure()
                    inline_fig.add_trace(go.Bar(x=mids, y=hist_vals, marker_color=color))
                    inline_fig.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center', 'font': {'size': 12}},
                                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                             margin=dict(l=8, r=6, t=24, b=6), xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
                    inline_card = html.Div([
                        dcc.Graph(figure=inline_fig, config={'displayModeBar': False}, style={'height': '120px', 'width': '140px'})
                    ], style={'flex': '0 0 140px', 'minWidth': '120px', 'padding': '6px', 'background': '#fff', 'borderRadius': '6px'})

                inline_graphs.append(inline_card)

            graphs_row = html.Div(inline_graphs, style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'justifyContent': 'center'})
            return graphs_row
        except Exception:
            # On any error, return empty so the UI doesn't break
            return []

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
    # Per-card info button: open auxiliary descriptors modal without overwriting main selection
    @app.callback(
        [Output('aux-selected-file-store', 'data', allow_duplicate=True), Output('aux-descriptors-open', 'data', allow_duplicate=True)],
        [Input({'type': 'show-aux-descriptors', 'filename': dash.dependencies.ALL, 'dataset': dash.dependencies.ALL}, 'n_clicks')],
        prevent_initial_call=True
    )
    def open_aux_descriptors(n_clicks_list):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update

        # find the first triggered pattern id with a truthy value
        triggered_prop = None
        for t in ctx.triggered:
            pid = t.get('prop_id', '')
            val = t.get('value')
            if 'show-aux-descriptors' in pid and val:
                triggered_prop = pid
                break

        if not triggered_prop:
            return no_update, no_update

        # extract JSON substring
        try:
            start = triggered_prop.find('{')
            end = triggered_prop.rfind('}')
            if start == -1 or end == -1:
                return no_update, no_update
            id_json = triggered_prop[start:end+1]
            try:
                payload = json.loads(id_json)
            except Exception:
                import ast
                try:
                    payload = ast.literal_eval(id_json)
                except Exception:
                    return no_update, no_update
        except Exception:
            return no_update, no_update

        filename = payload.get('filename')
        dataset = payload.get('dataset')

        try:
            df = get_cached_dataset_data(dataset) if dataset else file_df
        except Exception:
            df = file_df

        if df is None or df.empty or not filename:
            return no_update, no_update

        matched = df[df['filename'] == filename]
        if matched.empty:
            return no_update, no_update

        try:
            row = matched.iloc[0].to_dict()
            import pathlib as _pathlib
            import pandas as _pd

            def _sanitize_value(v):
                try:
                    if v is None:
                        return None
                    try:
                        if _pd.isna(v):
                            return None
                    except Exception:
                        pass
                    if isinstance(v, (_pathlib.Path, os.PathLike)):
                        return str(v)
                    if isinstance(v, (np.generic,)):
                        return v.item()
                    if isinstance(v, (np.ndarray,)):
                        return v.tolist()
                    if isinstance(v, _pd.Timestamp):
                        return v.isoformat()
                    if isinstance(v, dict):
                        return {str(k): _sanitize_value(val) for k, val in v.items()}
                    if isinstance(v, (list, tuple, set)):
                        return [_sanitize_value(x) for x in v]
                    if isinstance(v, (str, int, float, bool)):
                        return v
                    return str(v)
                except Exception:
                    return str(v)

            sanitized = {k: _sanitize_value(v) for k, v in row.items()}
            return sanitized, True
        except Exception as e:
            print(f"[DEBUG] open_aux_descriptors failed to prepare row: {e}")
            return no_update, no_update

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
        [Output('display-step-panel', 'style'), Output('center-action-buttons', 'style'), Output('inline-global-descriptors', 'style')],
        Input('selected-dataset-store', 'data')
    )
    def update_step_panel_visibility(selected_dataset):
        """
        Show/hide the step panel and center action buttons based on dataset type.
        Only show for datasets that contain processed step files.
        """
        if selected_dataset and ('UnifiedPreprocessed' in selected_dataset or 'Normalized' in selected_dataset):
            visible = {'display': 'block'}
            center_style = {'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '8px'}
            inline_style = {'display': 'flex', 'gap': '12px', 'alignItems': 'flex-start'}
        else:
            visible = {'display': 'none'}
            center_style = {'display': 'none'}
            inline_style = {'display': 'none'}

        return visible, center_style, inline_style

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
            return True, 6, 6, "Not available for this dataset", "step-info-text"
            
        if selected_file_data is None:
            return True, 6, 6, "Select a processed shape to enable step navigation", "step-info-text"
        
        # Extract filename from the selection data
        if isinstance(selected_file_data, dict):
            selected_filename = selected_file_data.get('filename')
            file_dataset = selected_file_data.get('dataset', selected_dataset)
        else:
            return True, 6, 6, "Invalid shape selection", "step-info-text"
        
        if not selected_filename:
            return True, 6, 6, "Invalid shape selection", "step-info-text"
        
        try:
            # Get the file data for the selected shape
            file_df = get_cached_dataset_data(file_dataset)
            if file_df is None or file_df.empty:
                return True, 6, 6, "Invalid shape selection", "step-info-text"
            
            # Find the file by filename
            matching_rows = file_df[file_df['filename'] == selected_filename]
            if matching_rows.empty:
                return True, 6, 6, "Shape not found in dataset", "step-info-text"
            
            row = matching_rows.iloc[0]
            
            # Check if shape has processing steps
            if not row.get('has_processing_steps', False):
                return True, 6, 6, "This shape has no processing steps available", "step-info-text"
            
            # Get available step information
            step_availability = get_available_steps(row)
            available_indices = step_availability['available_step_indices']
            recommended_max = step_availability['recommended_max_step']
            
            if not available_indices:
                return True, 6, 6, "No processing steps found for this shape", "step-info-text"
            
            # Set slider max to the highest available step
            slider_max = max(available_indices)
            
            # Set initial value to the recommended max step (usually the final step)
            initial_value = recommended_max
            
            # Create info message showing available steps
            missing_steps = step_availability['missing_step_indices']
            if missing_steps:
                step_names = ["Orig", "Mesh", "Trans", "Align", "Flip", "Scale", "Final"]
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
            return True, 6, 6, "Error checking processing steps", "step-info-text"

    @app.callback(
        [Output(f'step-label-{i}', 'className') for i in range(7)],
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
                if file_df is not None and file_df.empty:
                    return True, 6, 6, "Invalid shape selection", "step-info-text"
                
                # Find the row with matching filename
                matching_rows = file_df[file_df['filename'] == filename]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    print(f"✅ Found matching row for {filename}")
                    
                    # Get available steps for this shape
                    from core.file_index import get_available_steps
                    available_steps_info = get_available_steps(row)
                    available_steps = available_steps_info.get('available_step_indices', [])
                    print(f"🟢 Available steps for {filename}: {available_steps}")

                    # Ensure all step files are included in the merge
                    if selected_dataset and 'UnifiedPreprocessed' in selected_dataset:
                        file_df = get_cached_dataset_data(selected_dataset)
                        if file_df is not None and not file_df.empty:
                            # Filter rows to include all steps
                            step_files = file_df[file_df['filename'].str.contains('_step')]
                            if not step_files.empty:
                                file_df = pd.concat([file_df, step_files]).drop_duplicates()

                    # Update slider and step info dynamically
                    class_names = []
                    for i in range(7):
                        if i not in available_steps:
                            class_names.append("step-label missing")
                        elif i == step_value:
                            class_names.append("step-label active")
                        else:
                            class_names.append("step-label")

                    return class_names
                    
            except Exception as e:
                print(f"❌ Error in step label processing: {e}")
                import traceback
                traceback.print_exc()
        
        # Default fallback - no missing steps styling
        class_names = []
        for i in range(7):
            if is_disabled:
                class_names.append("step-label disabled")
            elif i == step_value:
                class_names.append("step-label active")
            else:
                class_names.append("step-label")
        
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

