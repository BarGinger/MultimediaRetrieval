from pathlib import Path
import dash
from dash import dcc, html, Input, Output
from core.file_index import get_file_tree
from .layout import build_layout
from .callbacks import register_callbacks

def discover_dataset_options():
    """
    Dynamically discover all valid dataset folders from the Datasets directory.
    A valid dataset folder contains category subdirectories with OBJ files.
    Also checks for nested datasets (e.g., UnifiedPreprocessed/Data).
    """
    # Find project root and datasets directory
    project_root = Path(__file__).resolve().parent.parent
    datasets_dir = project_root.parent.parent / "Datasets"
    
    if not datasets_dir.exists():
        print(f"⚠️ Datasets directory not found: {datasets_dir}")
        return ['Data']  # Fallback
    
    valid_datasets = []
    
    def check_for_valid_categories(dataset_path):
        """Check if a path contains valid category folders with OBJ files"""
        for category_path in dataset_path.iterdir():
            if category_path.is_dir() and not category_path.name.startswith('.'):
                # Check if category has OBJ files
                obj_files = list(category_path.glob("*.obj"))
                if obj_files:
                    return True
        return False
    
    # Check each subdirectory in Datasets
    for dataset_path in datasets_dir.iterdir():
        if dataset_path.is_dir() and not dataset_path.name.startswith('.'):
            
            # Check if this is a direct dataset (has categories directly)
            if check_for_valid_categories(dataset_path):
                valid_datasets.append(dataset_path.name)
            
            # Also check for nested datasets (e.g., UnifiedPreprocessed/Data)
            else:
                for nested_path in dataset_path.iterdir():
                    if nested_path.is_dir() and not nested_path.name.startswith('.'):
                        if check_for_valid_categories(nested_path):
                            # Use format: "ParentFolder/NestedFolder"
                            nested_name = f"{dataset_path.name}/{nested_path.name}"
                            valid_datasets.append(nested_name)
    
    # Sort datasets for consistent ordering
    valid_datasets.sort()
    
    # Ensure we have at least one dataset
    if not valid_datasets:
        print("⚠️ No valid datasets found, using fallback")
        valid_datasets = ['Data']
    
    print(f"📁 Discovered datasets: {valid_datasets}")
    return valid_datasets

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
    DATASET_OPTIONS = discover_dataset_options()
    DEFAULT_DATASET = DATASET_OPTIONS[0] if DATASET_OPTIONS else 'Data'
    
    try:
        file_df = get_file_tree(DEFAULT_DATASET)
    except Exception as e:
        print(f"⚠️ Error loading default dataset '{DEFAULT_DATASET}': {e}")
        # Try to find any working dataset
        file_df = None
        for dataset in DATASET_OPTIONS:
            try:
                file_df = get_file_tree(dataset)
                DEFAULT_DATASET = dataset
                break
            except Exception:
                continue
        
        if file_df is None:
            print("❌ No working datasets found")
            file_df = get_file_tree('Data')  # Final fallback

    # Base stores / top-level layout pieces
    app.layout = html.Div([
        dcc.Store(id="selected-file-store"),
        dcc.Store(id="selected-dataset-store", data=DEFAULT_DATASET),
        build_layout(file_df, DATASET_OPTIONS, DEFAULT_DATASET)
    ], style={'fontFamily': 'Arial, sans-serif'})

    # Register server callbacks
    register_callbacks(app, file_df, DATASET_OPTIONS, DEFAULT_DATASET)
    
    return app
