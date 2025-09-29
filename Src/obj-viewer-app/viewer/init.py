from pathlib import Path
import dash
from dash import dcc, html, Input, Output
from core.file_index import get_file_tree
from .layout import build_layout
from .callbacks import register_callbacks

def create_dash_app():
    # project_root points to the folder containing app.py and assets/
    project_root = Path(__file__).resolve().parent.parent

    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="3D Shape Viewer",
        assets_folder=str(project_root / "assets")
    )

    # --- Dataset options ---
    DATASET_OPTIONS = ['Data', 'Data_sampled', 'Data_sampled_resampled', 'Data_sampled_resampled_normalized', 'NormalizedShapes']
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
    
    # Add callback to handle normalization toggle based on dataset
    @app.callback(
        [Output('normalization-toggle', 'options'),
         Output('normalization-toggle', 'value')],
        [Input('selected-dataset-store', 'data')],
        prevent_initial_call=True
    )
    def update_normalization_toggle(selected_dataset):
        """Update normalization toggle based on selected dataset"""
        if selected_dataset == 'NormalizedShapes':
            # For NormalizedShapes, disable the toggle with explanation
            return [{'label': ' Already Normalized Dataset', 'value': 'normalized', 'disabled': True}], []
        else:
            # For other datasets, show normal toggle
            return [{'label': ' Show Normalized Shape', 'value': 'normalized'}], []
    
    return app
