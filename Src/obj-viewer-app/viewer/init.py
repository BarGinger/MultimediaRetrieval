from pathlib import Path
import dash
from dash import dcc, html
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

    file_df = get_file_tree()

    # Base stores / top-level layout pieces
    app.layout = html.Div([
        dcc.Store(id="selected-file-store"),
        build_layout(file_df)
    ], style={'fontFamily': 'Arial, sans-serif'})

    # Register server callbacks
    register_callbacks(app, file_df)
    return app
