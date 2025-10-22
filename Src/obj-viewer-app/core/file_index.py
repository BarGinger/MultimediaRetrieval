from pathlib import Path
import pandas as pd
import re

def get_file_tree(data_dir: str = "Data") -> pd.DataFrame:
    """
    Return DataFrame with columns: category, filename, filepath, size.
    Now enhanced to handle step-by-step processed files intelligently.
    """
    return get_file_tree_with_steps(data_dir)


def detect_step_files(shape_directory: Path, base_filename: str) -> dict:
    """
    Detect if a shape has step-by-step processing files available.
    
    Args:
        shape_directory: Path to the directory containing shape files
        base_filename: Base filename without extension (e.g., 'm1337')
    
    Returns:
        dict: Dictionary with step information:
        - 'has_steps': bool - whether step files are available
        - 'steps': dict - mapping of step names to file paths
        - 'original_file': str - path to the original file to display in list
    """
    step_pattern = rf"{re.escape(base_filename)}_(\d{{2}})_(\w+)\.obj$"
    step_files = {}
    
    # Look for step files in the directory
    for obj_file in shape_directory.glob(f"{base_filename}_*.obj"):
        match = re.match(step_pattern, obj_file.name)
        if match:
            step_num, step_name = match.groups()
            step_files[f"{step_num}_{step_name}"] = obj_file
    
    # Expected step files in order
    expected_steps = [
        "00_original",
        "01_remeshed", 
        "02_translated",
        "03_aligned", 
        "04_flipped",
        "05_scaled",
        "06_fill_holes_and_orientation"
    ]
    
    # Check if we have the key step files
    has_steps = any(step in step_files for step in expected_steps)
    
    # Determine which file to show in the main list
    # First check for unified file (final normalized result)
    unified_file = shape_directory / f"{base_filename}_unified.obj"
    if unified_file.exists():
        # Show the unified version in the list
        original_file = unified_file
    elif "05_scaled" in step_files:
        # Fallback to scaled version if no unified file
        original_file = step_files["05_scaled"]
    elif "06_fill_holes_and_orientation" in step_files:
        # Fallback to final step if available
        original_file = step_files["06_fill_holes_and_orientation"]
    elif has_steps:
        # If we have some steps but not the final one, show the highest numbered step
        available_steps = sorted([step for step in expected_steps if step in step_files])
        original_file = step_files[available_steps[-1]] if available_steps else None
    else:
        # No step files found, look for regular file
        regular_file = shape_directory / f"{base_filename}.obj"
        original_file = regular_file if regular_file.exists() else None
    
    return {
        'has_steps': has_steps,
        'steps': step_files,
        'original_file': str(original_file) if original_file else None,
        'step_count': len([step for step in expected_steps if step in step_files])
    }


def get_file_tree_with_steps(data_dir: str = "Data") -> pd.DataFrame:
    """
    Enhanced file tree that handles step-by-step processed files.
    Only shows _unified files (or _scaled files for unprocessed shapes) in the main list.
    
    Returns DataFrame with additional columns:
    - has_processing_steps: bool - whether this shape has step files
    - available_steps: int - number of available step files
    - step_files: dict - mapping of available step files
    """
    files_data = []
    cwd = Path.cwd()

    # Look for data_dir within Datasets folder
    dataset_path = f"Datasets/{data_dir}"
    candidates = [cwd / dataset_path, cwd.parent / dataset_path, cwd.parent.parent / dataset_path]
    data_path = next((p for p in candidates if p.exists()), candidates[0])

    if data_path.exists():
        # Special handling for NormalizedShapes dataset which has nested structure
        if data_dir == "NormalizedShapes":
            # Look for subdatasets within NormalizedShapes
            for subdataset_dir in data_path.iterdir():
                if subdataset_dir.is_dir():
                    # Each subdataset contains category directories
                    for category_dir in subdataset_dir.iterdir():
                        if category_dir.is_dir():
                            category = category_dir.name
                            _process_category_for_steps(category_dir, category, files_data)
        else:
            # Normal structure: Datasets/DataName/CategoryName/*.obj
            for category_dir in data_path.iterdir():
                if category_dir.is_dir():
                    category = category_dir.name
                    _process_category_for_steps(category_dir, category, files_data)

    df = pd.DataFrame(files_data)
    return df


def _process_category_for_steps(category_dir: Path, category: str, files_data: list):
    """
    Process a category directory to detect step files and add appropriate entries.
    
    Args:
        category_dir: Path to category directory
        category: Category name
        files_data: List to append file data to
    """
    # Track processed base names to avoid duplicates
    processed_shapes = set()
    
    # First pass: find all step files and group by base name
    shape_groups = {}
    
    for obj_file in category_dir.glob("*.obj"):
        # Extract base filename
        filename = obj_file.name
            
        # Check if this is a step file or unified file
        step_match = re.match(r"(.+)_\d{2}_\w+\.obj$", filename)
        unified_match = re.match(r"(.+)_unified\.obj$", filename)
        
        if step_match:
            base_name = step_match.group(1)
            if base_name not in shape_groups:
                shape_groups[base_name] = []
            shape_groups[base_name].append(obj_file)
        elif unified_match:
            base_name = unified_match.group(1)
            if base_name not in shape_groups:
                shape_groups[base_name] = []
            shape_groups[base_name].append(obj_file)
        else:
            # Regular file (no step suffix) - only include if no step files exist for this base
            base_name = obj_file.stem  # filename without extension
            if base_name not in shape_groups:
                shape_groups[base_name] = []
            shape_groups[base_name].append(obj_file)
    
    # Second pass: for each shape group, determine what to show in the list
    for base_name, obj_files in shape_groups.items():
        if base_name in processed_shapes:
            continue
            
        step_info = detect_step_files(category_dir, base_name)
        
        # Determine which file to use for the main listing
        display_file = None
        if step_info['has_steps'] and step_info['original_file']:
            # Prefer step files - show the final processed version
            display_file = Path(step_info['original_file'])
        elif obj_files:
            # Fallback to regular file if no step files
            display_file = obj_files[0]
        
        if display_file and display_file.exists():
            files_data.append({
                "category": category,
                "filename": display_file.name,
                "filepath": str(display_file),
                "size": display_file.stat().st_size,
                "has_processing_steps": step_info['has_steps'],
                "available_steps": step_info['step_count'],
                "step_files": step_info['steps'],
                "base_filename": base_name
            })
            processed_shapes.add(base_name)


def get_step_file_path(row: pd.Series, step_index: int = 5) -> tuple:
    """
    Get the file path for a specific processing step.
    
    Args:
        row: DataFrame row containing shape information
        step_index: Step index (0-5):
            0: 00_original
            1: 01_remeshed (if available)  
            2: 02_translated
            3: 03_aligned
            4: 04_flipped
            5: 05_scaled (default - final result)
            6: 06_fill_holes_and_orientation (if available)
    
    Returns:
        tuple: (file_path, actual_step_index, step_info)
            - file_path: Path to the step file
            - actual_step_index: The actual step index found (may differ from requested)
            - step_info: Dict with step information including availability
    """
    # Check if this shape has processing steps
    if not row.get('has_processing_steps', False):
        # No processing steps, return original file
        step_info = get_step_display_info(0)
        step_info['requested_step'] = step_index
        step_info['actual_step'] = 0
        step_info['step_available'] = True
        step_info['fallback_used'] = step_index != 0
        return row['filepath'], 0, step_info
    
    # Step mapping
    step_names = [
        "00_original",
        "01_remeshed", 
        "02_translated",
        "03_aligned",
        "04_flipped", 
        "05_scaled",
        "06_fill_holes_and_orientation"
    ]
    
    if step_index < 0 or step_index >= len(step_names):
        step_index = 5  # Default to final step
    
    requested_step = step_names[step_index]
    step_files = row.get('step_files', {})
    
    # Try to get the requested step file
    if requested_step in step_files:
        step_info = get_step_display_info(step_index)
        step_info['requested_step'] = step_index
        step_info['actual_step'] = step_index
        step_info['step_available'] = True
        step_info['fallback_used'] = False
        return str(step_files[requested_step]), step_index, step_info
    
    # Fallback: find the closest available step
    actual_step_index = None
    actual_file_path = None
    
    # First try steps before the requested step (reverse order to find closest previous)
    for i in range(step_index - 1, -1, -1):
        if step_names[i] in step_files:
            actual_step_index = i
            actual_file_path = str(step_files[step_names[i]])
            break
    
    # If not found, try steps after the requested step
    if actual_step_index is None:
        for i in range(step_index, len(step_names)):
            if step_names[i] in step_files:
                actual_step_index = i
                actual_file_path = str(step_files[step_names[i]])
                break
    
    # If still not found, use original file
    if actual_step_index is None:
        actual_step_index = 0
        actual_file_path = row['filepath']
    
    # Create step info with fallback details
    step_info = get_step_display_info(actual_step_index)
    step_info['requested_step'] = step_index
    step_info['actual_step'] = actual_step_index
    step_info['step_available'] = False
    step_info['fallback_used'] = True
    step_info['requested_step_name'] = get_step_display_info(step_index)['name']
    
    return actual_file_path, actual_step_index, step_info


def get_step_display_info(step_index: int = 5) -> dict:
    """
    Get display information for a processing step.
    
    Args:
        step_index: Step index (0-5)
    
    Returns:
        dict: Information about the step including name, description, etc.
    """
    step_info = [
        {"name": "Original", "description": "Original imported shape", "color": "#95a5a6"},
        {"name": "Remeshed", "description": "Resampled mesh (target ~7500 vertices)", "color": "#3498db"},
        {"name": "Translated", "description": "Centered at origin (barycenter)", "color": "#9b59b6"},
        {"name": "Aligned", "description": "PCA aligned to axes", "color": "#e67e22"},
        {"name": "Flipped", "description": "Consistent orientation (moment test)", "color": "#f39c12"},
        {"name": "Scaled", "description": "Unit bounding box", "color": "#27ae60"},
        {"name": "Fill Holes & Orientation", "description": "Holes filled and correct orientation", "color": "#0c6f43"},
    ]
    
    if step_index < 0 or step_index >= len(step_info):
        step_index = 5
    
    return step_info[step_index]


def get_available_steps(row: pd.Series) -> dict:
    """
    Get information about which processing steps are available for a shape.
    
    Args:
        row: DataFrame row containing shape information
    
    Returns:
        dict: Information about available steps
    """
    result = {
        'has_processing_steps': row.get('has_processing_steps', False),
        'available_step_indices': [],
        'missing_step_indices': [],
        'step_availability': {},
        'recommended_max_step': 0
    }
    
    if not result['has_processing_steps']:
        # Only original step available
        result['available_step_indices'] = [0]
        result['step_availability'] = {0: True, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False}
        result['recommended_max_step'] = 0
        return result
    
    step_names = [
        "00_original",
        "01_remeshed", 
        "02_translated",
        "03_aligned",
        "04_flipped", 
        "05_scaled",
        "06_fill_holes_and_orientation"
    ]
    
    step_files = row.get('step_files', {})
    
    # Check each step's availability
    for i, step_name in enumerate(step_names):
        is_available = step_name in step_files
        result['step_availability'][i] = is_available
        
        if is_available:
            result['available_step_indices'].append(i)
            result['recommended_max_step'] = i  # Keep updating to get the highest available
        else:
            result['missing_step_indices'].append(i)
    
    return result
