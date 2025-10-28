from pathlib import Path
import dash
from dash import dcc, html, Input, Output
from core.file_index import get_file_tree
from core.dataset_cache import get_available_datasets, get_cached_dataset_data, preload_datasets
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

    print("🚀 Initializing high-performance 3D Shape Viewer...")
    
    # Use high-performance dataset discovery
    print("📁 Discovering available datasets...")
    DATASET_OPTIONS = get_available_datasets()
    # Prefer the UnifiedPreprocessed/Data dataset when available (either discovered or present on disk)
    preferred = 'UnifiedPreprocessed/Data'
    if preferred in DATASET_OPTIONS:
        DEFAULT_DATASET = preferred
    else:
        # If dataset discovery didn't return the preferred name, check for the on-disk path
        preferred_path = project_root.parent / 'Datasets' / 'UnifiedPreprocessed' / 'Data'
        if preferred_path.is_dir():
            DEFAULT_DATASET = preferred
        else:
            DEFAULT_DATASET = DATASET_OPTIONS[0] if DATASET_OPTIONS else 'Data'
    print(f"✅ Found {len(DATASET_OPTIONS)} datasets: {DATASET_OPTIONS}")
    
    # Preload all datasets for instant switching
    print("⚡ Preloading datasets for instant switching...")
    preload_datasets()
    print("✅ All datasets preloaded!")
    
    # Get initial file data (now instant from cache)
    file_df = get_cached_dataset_data(DEFAULT_DATASET)
    print(f"🎯 Using default dataset: {DEFAULT_DATASET} ({len(file_df)} shapes)")

    # Base stores / top-level layout pieces
    app.layout = html.Div([
        dcc.Store(id="selected-file-store"),
        dcc.Store(id="selected-dataset-store", data=DEFAULT_DATASET),
    # Ensure a stable hidden Close button exists at top-level so callbacks referencing
    # its `n_clicks` property validate correctly (prevents missing-input errors).
    build_layout(file_df, DATASET_OPTIONS, DEFAULT_DATASET)
    ], style={'fontFamily': 'Arial, sans-serif'})

    # Register server callbacks
    register_callbacks(app, file_df, DATASET_OPTIONS, DEFAULT_DATASET)
    
    return app
