import dash
from dash import dcc, html
from core.file_index import get_file_tree
from .layout import build_layout
from .callbacks import register_callbacks

def create_dash_app():
    app = dash.Dash(__name__, suppress_callback_exceptions=True,
                    title="3D Shape Viewer")

    # --- Dataset options ---
    DATASET_OPTIONS = ['Data', 'Data_sampled', 'Data_sampled_resampled', 'Data_sampled_resampled_normalized']
    DEFAULT_DATASET = DATASET_OPTIONS[1]
    file_df = get_file_tree(DEFAULT_DATASET)

    # Base stores / top-level layout pieces
    app.layout = html.Div([
        dcc.Store(id="selected-file-store"),
        dcc.Store(id="selected-dataset-store", data=DEFAULT_DATASET),
        build_layout(file_df, DATASET_OPTIONS, DEFAULT_DATASET)
    ], style={'fontFamily': 'Arial, sans-serif'})

    # Register server callbacks
    register_callbacks(app, file_df, DATASET_OPTIONS, DEFAULT_DATASET)
    return app
