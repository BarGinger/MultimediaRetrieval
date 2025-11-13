# 3D Shape Viewer Application

## Overview
The 3D Shape Viewer Application is a web-based tool built with Dash/Plotly that allows users to browse, visualize, and analyze 3D mesh models from the Princeton Shape Benchmark (PSB) dataset. The application provides interactive 3D visualization, feature extraction, shape similarity search, and comprehensive descriptor analysis.

## Features
- **Interactive 3D Visualization**: View and rotate 3D mesh models in real-time
- **Multi-Dataset Support**: Browse different preprocessing stages (Data, Data_sampled, Jet, UnifiedPreprocessed)
- **Shape Analysis**: View geometric descriptors and statistical properties for each shape
- **Similarity Search**: Find similar shapes using K-nearest neighbors with customizable distance metrics
- **Descriptor Visualization**: Interactive histograms and distribution plots for all shape features
- **Scalability Visualization**: View dimensionality reduction embeddings (t-SNE) of the shape space
- **Category-based Organization**: Browse shapes by their semantic categories with color coding
- **Multi-step Processing Pipeline**: View shapes at different preprocessing stages (original, normalized, resampled, aligned)

## Project Structure
```
obj-viewer-app/
├── app.py                          # Application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── viewer/                         # Main Dash application
│   ├── init.py                     # App initialization and dataset preloading
│   ├── layout.py                   # UI layout definition
│   ├── callbacks.py                # All Dash callbacks and interactivity
│   └── category_colors.py          # Category color definitions
│
├── core/                           # Core functionality modules
│   ├── obj_parser.py               # OBJ file parser
│   ├── shapeMesh.py                # Mesh data structure and operations
│   ├── transformations.py          # Geometric transformations
│   ├── extractions.py              # Feature extraction (scalar descriptors)
│   ├── extractions_minmax.py       # Feature extraction with min/max normalization
│   ├── dataset_cache.py            # Dataset caching system
│   ├── file_index.py               # File tree indexing
│   ├── normalized_cache.py         # Cached normalized features
│   ├── plotting.py                 # 3D plotting utilities
│   └── analysis_cache.py           # Analysis results caching
│
├── assets/                         # Static web assets
│   ├── style.css                   # Custom CSS styling
│   ├── clientsize.js               # Client-side viewport detection
│   └── modal_close_proxy.js        # Modal interaction helper
│
├── .dataset_cache/                 # Cached dataset metadata (auto-generated)
│
├── normelization/                  # Preprocessing and validation scripts
│   ├── normalize_database.py       # Database normalization
│   ├── evaluate_preprocessing.py   # Validation and metrics
│   ├── generate_validation_plots.py # Visualization generation
│   └── ...                         # Additional preprocessing utilities
│
├── evalution/                      # Evaluation results and metrics
│   ├── combined_overall_summary.csv
│   ├── combined_per_class_summary.csv
│   └── matrix_*/                   # Distance matrix evaluations
│
└── Utility scripts:
    ├── extract_features.py         # Batch feature extraction
    ├── extract_histogram_values.py # Batch histogram value extraction
    ├── find_percentile.py          # Percentile-based normalization
    ├── find_minmax.py              # Min/max aggregation
    ├── merge_csvs.py               # CSV merging utilities
    └── create_prepared_files.py    # Preprocessing pipeline
```

## Required CSV Files and Data Structure

The application requires specific CSV files to be placed in the correct locations:

### 1. Analysis Results (Required)
**Location**: `Datasets/UnifiedPreprocessed/Data/analysis_results_unifiedPreprocessed_data.csv`

This file contains all extracted features and descriptors for each 3D shape. It should have columns:
- `file`: Shape filename (e.g., `m0.obj`)
- `class`: Category/class name
- Shape descriptors: `surface_area`, `compactness`, `rectangularity`, `diameter`, `convexity`, `eccentricity`
- Distribution features: `A3_*`, `D1_*`, `D2_*`, `D3_*`, `D4_*` (histogram bins)

**Generation**: Run `extract_features.py` followed by appropriate normalization scripts

### 2. Distance Matrix (Required for Similarity Search)
**Location**: `Src/matching/matrix_rank_based_optimized.csv`

This file contains precomputed pairwise distances between all shapes in the database.
- Format: CSV with shape filenames as both row and column indices
- Values: Distance/similarity scores between shape pairs

**Generation**: Computed by the matching/similarity module using normalized features

### 3. Scalability Visualization Files (for t-SNE dialog)
**Location**: 
- `Src/scalability/topology_graph.csv` - 2D/3D embeddings
- `Src/scalability/class_labels.csv` - Class labels for each shape

**Format**:
- `topology_graph.csv`: Columns `file`, `x`, `y` (and optionally `z`)
- `class_labels.csv`: Columns `file`, `class`

**Generation**: Run dimensionality reduction (t-SNE/UMAP) on normalized feature vectors

### 4. 3D Mesh Files (Required)
**Base Location**: `Datasets/`

The application expects mesh files organized by dataset:
```
Datasets/
├── Data/                    # Original PSB dataset
│   ├── Airplane/
│   │   ├── m0.obj
│   │   ├── m1.obj
│   │   └── ...
│   ├── Ant/
│   └── ...
│
├── UnifiedPreprocessed/     # Preprocessed meshes
│   ├── Data/                # Main normalized dataset
│   │   ├── Airplane/
│   │   └── ...
│   ├── Data_sampled/        # Resampled meshes
│   └── Jet/                 # Specific category subset
│
└── [Other dataset variants]
```

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup Steps

1. **Clone/Download the repository** and navigate to the obj-viewer-app directory:
   ```bash
   cd obj-viewer-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare required data files**:
   - Ensure 3D mesh files (.obj) are in `Datasets/` subdirectories
   - Generate or place `analysis_results_unifiedPreprocessed_data.csv` in `Datasets/UnifiedPreprocessed/Data/`
   - Place distance matrix in `Src/matching/matrix_rank_based_optimized.csv`
   - Place scalability CSVs in `Src/scalability/`

4. **Verify data structure**:
   ```bash
   # Check that analysis results exist
   ls Datasets/UnifiedPreprocessed/Data/analysis_results_unifiedPreprocessed_data.csv
   
   # Check that mesh files exist
   ls Datasets/UnifiedPreprocessed/Data/Airplane/
   ```

## Usage

### Starting the Application

Run the application from the `obj-viewer-app` directory:

```bash
python app.py
```

The application will:
1. Preload dataset metadata and cache it for faster subsequent loads
2. Start a local web server on `http://127.0.0.1:8050`
3. Print status messages to the console

Open your web browser and navigate to: **http://127.0.0.1:8050**

### Using the Interface

1. **Select a Dataset**: Choose from available datasets (Data, Data_sampled, Jet, UnifiedPreprocessed)
2. **Browse Categories**: Click on categories in the left panel to filter shapes
3. **Select a Shape**: Click on a shape thumbnail to view it in 3D
4. **View Processing Steps**: Use the dropdown to see different preprocessing stages
5. **Analyze Descriptors**: Click "Show Shape Info" to see detailed feature values and histograms
6. **Find Similar Shapes**: Click "Find Similar Shapes" to see K-nearest neighbors
7. **Explore Scalability**: Use the scalability tab to view dimensionality reduction embeddings

### Configuration

Key settings in the code:
- `app.py`: Port and host configuration (default: 127.0.0.1:8050)
- `viewer/callbacks.py`: Number of similar shapes to retrieve (K value)
- `core/dataset_cache.py`: Dataset paths and caching behavior

## Feature Extraction Pipeline

To generate the required CSV files from raw mesh data:

1. **Extract Features**:
   ```bash
   python extract_features.py --dataset_path "Datasets/UnifiedPreprocessed/Data" --output "analysis_results.csv"
   ```

2. **Compute Histogram Values**:
   ```bash
   python extract_histogram_values.py
   ```

3. **Normalize Features** (using preprocessing scripts in `normelization/`):
   ```bash
   cd normelization
   python normalize_database.py
   ```

4. **Generate Distance Matrix** (from the matching module):
   ```bash
   cd ../matching
   python compute_distances.py
   ```

## Development Notes

- **Caching**: Dataset metadata is cached in `.dataset_cache/` for faster loading
- **Performance**: Large datasets may take time to load initially; subsequent loads use cached data
- **Debugging**: Set `debug=True` in `app.py` for auto-reload during development
- **Extensions**: Add new callbacks in `viewer/callbacks.py`, new features in `core/extractions.py`

## Troubleshooting

### Common Issues

1. **"File not found" errors**:
   - Verify CSV file paths match the structure above
   - Check file path casing (Windows is case-insensitive, Linux/Mac are case-sensitive)

2. **Empty shape list**:
   - Ensure `analysis_results_unifiedPreprocessed_data.csv` exists and contains data
   - Verify mesh files exist in the specified dataset directory

3. **Similarity search not working**:
   - Ensure `matrix_rank_based_optimized.csv` exists in `Src/matching/`
   - Verify the distance matrix has matching shape names

4. **Slow loading**:
   - First load caches metadata and may be slow for large datasets
   - Subsequent loads should be faster
   - Consider using a smaller dataset for testing

## Technical Stack

- **Frontend**: Dash (Plotly), HTML/CSS, JavaScript
- **Backend**: Python, Flask (via Dash)
- **Visualization**: Plotly.js for 3D graphics
- **Data Processing**: pandas, NumPy
- **Similarity Search**: scikit-learn (KDTree), precomputed distance matrices
- **Caching**: JSON-based metadata cache
