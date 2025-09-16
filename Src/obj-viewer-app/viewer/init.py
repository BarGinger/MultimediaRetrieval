import dash
from dash import dcc, html
from core.file_index import get_file_tree
from .layout import build_layout
from .callbacks import register_callbacks

def create_dash_app():
    app = dash.Dash(__name__, suppress_callback_exceptions=True,
                    title="3D Shape Viewer")

    # --- Toggle for sampled dataset ---
    USE_SAMPLED_DATASET = True  # Set to True to use Data_sampled, False for full Data
    data_dir = 'Data_sampled' if USE_SAMPLED_DATASET else 'Data'
    file_df = get_file_tree(data_dir)

    # Base stores / top-level layout pieces
    app.layout = html.Div([
        dcc.Store(id="selected-file-store"),
        build_layout(file_df)
    ], style={'fontFamily': 'Arial, sans-serif'})

    # Register server callbacks
    register_callbacks(app, file_df, USE_SAMPLED_DATASET)
    return app
