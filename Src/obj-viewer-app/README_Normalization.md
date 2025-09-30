# Step 3.1: Full Normalization with Caching

This implementation provides an efficient cached normalization system for 3D shapes, following the 4-step normalization pipeline described in the course technical tips.

## 🚀 Quick Start

### 1. Run Preprocessing (One-time setup)
```bash
cd Src/obj-viewer-app/preprocessing
python normalize_database.py
```

This will:
- Process all shapes in your datasets
- Apply the 4-step normalization pipeline
- Save normalized shapes as OBJ files
- Generate metadata and processing reports

### 2. Check Preprocessing Status
```bash
cd Src/obj-viewer-app
python check_preprocessing.py
```

### 3. Test Normalization
```bash
cd Src/obj-viewer-app
python test_normalization.py
```

### 4. Use the Viewer with Normalization
```bash
cd Src/obj-viewer-app
python app.py
```

Then toggle "Show Normalized Shape" in the UI to compare original vs normalized shapes.

## 📋 4-Step Normalization Pipeline

### Step 1: Translation (Centering)
- Moves the shape's barycenter to the origin (0,0,0)
- Formula: `vertices_centered = vertices - barycenter`

### Step 2: Alignment (PCA)
- Aligns principal axes with coordinate frame using PCA eigenvectors
- Uses the formulas from technical tips:
  - `x_updated = (p_i - c) · e1`
  - `y_updated = (p_i - c) · e2`
  - `z_updated = (p_i - c) · (e1 × e2)`

### Step 3: Flipping (Moment Test)
- Ensures consistent orientation using moment test
- Formula: `f_i = Σ sign(C_t,i) * (C_t,i)^2`
- Flips along axis i if `f_i < 0`

### Step 4: Scaling
- Scales shape to fit in unit bounding box
- Formula: `vertices_scaled = vertices / max_dimension`

## 🎯 Benefits of Cached Approach

1. **⚡ Performance**: Normalization computed once, reused thousands of times
2. **🎯 Consistency**: Exact same normalized shapes used for visualization and future feature extraction
3. **💾 Storage**: Normalized shapes saved as standard OBJ files (reusable by other tools)
4. **📈 Scalability**: Process entire database overnight, use instantly during development
5. **🔍 Debugging**: Detailed metadata files contain normalization information
6. **🚀 Ready for Step 3.2**: Normalized shapes ready for feature extraction

## 📊 File Structure

```
NormalizedShapes/
├── LabeledPSB/
│   ├── shape1_normalized.obj      # Normalized mesh vertices and faces
│   ├── shape1_metadata.json       # PCA info, flipping factors, etc.
│   ├── shape2_normalized.obj
│   ├── shape2_metadata.json
│   └── ...
├── princeton/
│   └── ...
└── normalization_report.json      # Processing statistics and quality metrics
```

## 🔧 API Usage

### Basic Normalization
```python
from core.shapeMesh import ShapeMesh

# Create mesh and normalize
mesh = ShapeMesh.from_file_row(row)
normalized_vertices = mesh.apply_full_normalization()

# Get detailed normalization info
norm_info = mesh.get_normalization_info()
```

### Cached Normalization
```python
from core.shapeMesh import ShapeMesh
from core.normalized_cache import normalized_cache

# Use cached version if available, compute otherwise
mesh = ShapeMesh.from_file_row(row, use_normalized=True, dataset="LabeledPSB")

# Or get just the normalized vertices efficiently
vertices = mesh.get_normalized_vertices_cached(dataset="LabeledPSB")
```

## 📈 Quality Verification

The system automatically verifies normalization quality:

- **Centering**: Center should be at origin (error < 1e-10)
- **Scaling**: Max dimension should be 1.0 (error < 1e-6)
- **Processing Report**: Contains statistics for entire database

Example quality output:
```
✅ Shape properly centered
✅ Shape properly scaled to unit size
✅ Normalization PASSED
```

## 🚀 Ready for Step 3.2

With normalized shapes cached, you're ready to:
1. Extract shape features (surface area, volume, etc.)
2. Implement shape descriptors
3. Build similarity search
4. Create shape retrieval system

The normalized shapes ensure consistent feature extraction across all shapes in your database!