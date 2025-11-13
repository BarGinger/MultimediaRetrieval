# 3D Shape Retrieval System
**Multimedia Retrieval Project - INFOMR**

## Project Goal

Build a content-based 3D shape retrieval system that, given a 3D shape, finds and shows the most similar shapes in a 3D shape database.

**Course:** Multimedia Retrieval (INFOMR) - Utrecht University  
**Assignment Details:** [https://webspace.science.uu.nl/~telea001/MR/Assignment](https://webspace.science.uu.nl/~telea001/MR/Assignment)

## Contributors

| Name | Email | Git User |
|------|-------|----------|
| Bar Melinarskiy | b.melinarskiy@students.uu.nl | BarGinger |
| Arie Klaver | a.c.klaver@students.uu.nl | Arie-K |
| Rutger Vincken | r.l.g.vincken@students.uu.nl | rutgervincken |

## Project Structure

```
MultimediaRetrieval/
├── Datasets/                           # 3D shape databases
│   ├── Data/                           # Original Princeton Shape Benchmark dataset
│   │   ├── Airplane/
│   │   ├── Ant/
│   │   ├── Armadillo/
│   │   └── ... (380 shape classes)
│   ├── Data_sampled/                   # Resampled mesh variants
│   ├── UnifiedPreprocessed/            # Preprocessed and normalized meshes
│   │   ├── Data/                       # Main normalized dataset
│   │   │   ├── analysis_results_unifiedPreprocessed_data.csv
│   │   │   ├── Airplane/
│   │   │   └── ...
│   │   ├── Data_sampled/
│   │   └── Jet/
│   └── FeatureExtractions/             # Extracted feature CSVs
│
├── Src/                                # Source code
│   ├── obj-viewer-app/                 # Main 3D Shape Viewer Application (Dash/Plotly)
│   │   ├── app.py                      # Application entry point
│   │   ├── requirements.txt            # Python dependencies
│   │   ├── README.md                   # Detailed app documentation
│   │   ├── viewer/                     # Dash UI components
│   │   │   ├── callbacks.py            # Interactive callbacks
│   │   │   ├── layout.py               # UI layout
│   │   │   └── category_colors.py      # Category styling
│   │   ├── core/                       # Core functionality
│   │   │   ├── obj_parser.py           # OBJ file parsing
│   │   │   ├── shapeMesh.py            # Mesh data structures
│   │   │   ├── transformations.py      # Geometric operations
│   │   │   ├── extractions.py          # Feature extraction
│   │   │   ├── dataset_cache.py        # Dataset caching
│   │   │   └── plotting.py             # 3D visualization
│   │   ├── normelization/              # Preprocessing pipeline
│   │   │   ├── normalize_database.py   # Database normalization
│   │   │   ├── evaluate_preprocessing.py # Validation
│   │   │   └── generate_validation_plots.py
│   │   ├── assets/                     # Web assets (CSS, JS)
│   │   └── evalution/                  # Evaluation metrics
│   │
│   ├── matching/                       # Shape similarity and retrieval
│   │   ├── shapeQuery.py               # Query processing
│   │   ├── shapeFeatures.py            # Feature computation
│   │   ├── distance.py                 # Distance metrics
│   │   ├── compute_distances.py        # Distance matrix generation
│   │   ├── matrix_rank_based_optimized.csv # Precomputed distances
│   │   ├── matrix_minmax_optimized.csv
│   │   ├── matrix_weighted_sum.csv
│   │   ├── optimize_*.py               # Weight optimization scripts
│   │   └── optimization_results_*/     # Optimization results
│   │
│   ├── scalability/                    # Dimensionality reduction and indexing
│   │   ├── build_kdtree.py             # KD-Tree construction
│   │   ├── topology_graph.py           # t-SNE/UMAP embeddings
│   │   ├── topology_graph.csv          # 2D/3D embeddings
│   │   ├── class_labels.csv            # Shape labels
│   │   ├── kdtree.joblib               # Serialized KD-Tree
│   │   └── Scalability.py              # Scalability analysis
│   │
│   └── evalution/                      # Evaluation and metrics
│       ├── evalution.py                # Main evaluation script
│       ├── create_query_visualization.py # Query result visualization
│       └── figures/                    # Evaluation plots and metrics
│           ├── combined_overall_summary.csv
│           ├── combined_per_class_summary.csv
│           ├── kdtree/
│           ├── matrix_minmax_optimized/
│           └── matrix_rank_based_optimized/
│
├── Preprocessing/                      # Data preprocessing utilities
│   ├── sample_dataset.py
│   ├── resampling_simple.py
│   ├── normalization.py
│   └── analyzer_tool.py
│
├── scripts/                            # Utility scripts
│   ├── merge.py
│   ├── standardize_features.py
│   └── find_nan_or_zero_rows.py
│
├── output/                             # Generated outputs and intermediate files
├── Reports/                            # Project reports and documentation
│   ├── Step1/                          # Step 1 deliverables
│   ├── Step3/                          # Step 3 deliverables
│   └── Step6/                          # Step 6 deliverables
│
└── README.md                           # This file
```

## How to Run

### Quick Start

1. **Navigate to the application directory**:
   ```bash
   cd Src/obj-viewer-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the 3D Shape Viewer**:
   ```bash
   python app.py
   ```

4. **Open in browser**: Navigate to `http://127.0.0.1:8050`

### Detailed Instructions

For comprehensive setup instructions, data requirements, and usage guide, see:
**[Src/obj-viewer-app/README.md](Src/obj-viewer-app/README.md)**

The application README covers:
- Required CSV files and their locations
- Feature extraction pipeline
- Dataset structure requirements
- Troubleshooting common issues
- Complete feature documentation

## Key Features

- **Interactive 3D Visualization**: Real-time mesh rendering with Plotly
- **Shape Similarity Search**: K-nearest neighbors using optimized distance metrics
- **Feature Extraction**: 
  - Global descriptors (surface area, compactness, rectangularity, etc.)
  - Shape distributions (A3, D1, D2, D3, D4 histograms)
- **Multi-stage Preprocessing**: Original → Normalized → Resampled → Aligned
- **Dimensionality Reduction**: t-SNE visualization of shape space
- **Performance Optimization**: KD-Tree indexing, precomputed distance matrices
- **Comprehensive Evaluation**: Per-class and overall retrieval metrics

## System Performance

See evaluation results in `Src/evalution/figures/`:
- Overall retrieval accuracy metrics
- Per-class performance analysis
- Confusion matrices and precision-recall curves
- Comparison of different distance metrics (rank-based, min-max, weighted sum)

---

**Last Updated:** November 2025
