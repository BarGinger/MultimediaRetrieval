import os
import numpy as np
import pandas as pd
from shapeFeatures import Shape


class ShapeDistance:
    """
    A class for computing distance metrics between 3D shapes.
    
    Supports various distance measures for histogram-based shape descriptors
    and global shape features.
    """
    
    def __init__(self, shape_a, shape_b, precomputed_dir: str | None = "distance_matrices_normalized_98", debug: bool = False):
        """
        Initialize ShapeDistance with two Shape objects.
        
        Args:
            shape_a (Shape): First shape to compare
            shape_b (Shape): Second shape to compare
            precomputed_dir (str | None): Directory containing precomputed distance matrices
                (CSV files) to look up distances. If None, always compute on the fly.
                If relative path, resolved relative to this file. Default: 'distance_matrices_normalized_98'.
            debug (bool): If True, print timing information for performance analysis
        """
        if not isinstance(shape_a, Shape) or not isinstance(shape_b, Shape):
            raise TypeError("Both arguments must be Shape objects")
        
        self.debug = debug
        self.shape_a = shape_a
        self.shape_b = shape_b
        
        # Descriptor sets for modularity
        self.HISTOGRAM_DESCRIPTORS = ['A3', 'D1', 'D2', 'D3', 'D4']
        self.GLOBAL_DESCRIPTORS = [
            'surface_area', 'compactness', 'rectangularity',
            'diameter', 'convexity', 'eccentricity'
        ]

        # Optional precomputed matrices directory and cache
        if precomputed_dir in (None, ""):
            self.precomputed_dir = None
            if self.debug:
                print("[DEBUG] No precomputed directory - will compute all distances on the fly")
        else:
            # Resolve relative to this script file
            if not os.path.isabs(precomputed_dir):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                precomputed_dir = os.path.join(script_dir, precomputed_dir)
            self.precomputed_dir = precomputed_dir
            if self.debug:
                print(f"[DEBUG] Precomputed directory: {self.precomputed_dir}")
                print(f"[DEBUG] Directory exists: {os.path.exists(self.precomputed_dir)}")
        self._matrix_cache: dict[str, pd.DataFrame] = {}

    # -----------------------------
    # Precomputed matrix helpers
    # -----------------------------
    def _load_matrix(self, descriptor_name: str, is_global: bool) -> pd.DataFrame | None:
        """Load and cache a precomputed distance matrix for a descriptor, if available."""
        if self.precomputed_dir is None:
            return None
        # Filename by convention
        fname = f"distances_global_{descriptor_name}.csv" if is_global else f"distances_{descriptor_name}.csv"
        path = os.path.join(self.precomputed_dir, fname)
        if path in self._matrix_cache:
            return self._matrix_cache[path]
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, index_col=0)
            self._matrix_cache[path] = df
            return df
        except Exception:
            return None

    def _lookup_precomputed(self, descriptor_name: str, is_global: bool) -> float | None:
        """Try to look up a precomputed distance for the current shape pair.
        Returns a float distance or None if not found/available."""
        if self.debug:
            import time
            t_start = time.time()
        
        df = self._load_matrix(descriptor_name, is_global)
        if df is None:
            if self.debug:
                print(f"[DEBUG]   {descriptor_name}: No precomputed matrix found")
            return None
        
        a = self.shape_a.shape
        b = self.shape_b.shape
        if a not in df.index or b not in df.columns:
            # Indices might be symmetric, try both orientations
            if b not in df.index or a not in df.columns:
                if self.debug:
                    print(f"[DEBUG]   {descriptor_name}: Shapes not in matrix index")
                return None
        # Try [a, b]
        val = df.at[a, b] if (a in df.index and b in df.columns) else np.nan
        # If NaN, try symmetric [b, a]
        if pd.isna(val):
            val = df.at[b, a] if (b in df.index and a in df.columns) else np.nan
        if pd.isna(val):
            # Diagonal case
            if a == b:
                return 0.0
            if self.debug:
                print(f"[DEBUG]   {descriptor_name}: Value is NaN in matrix")
            return None
        try:
            result = float(val)
            if self.debug:
                t_end = time.time()
                print(f"[DEBUG]   {descriptor_name}: Lookup succeeded ({(t_end-t_start)*1000:.2f}ms) = {result:.6f}")
            return result
        except Exception:
            return None
    
    def histogram_distance(self, descriptor_name):
        """
        Compare histograms of a specific descriptor between two shapes using EMD.
        
        Args:
            descriptor_name (str): Name of the descriptor ('A3', 'D1', 'D2', 'D3', 'D4')
        
        Returns:
            float: EMD (Earth Mover's Distance) between the two histograms
            
        Raises:
            ValueError: If descriptor_name is not valid or histograms are missing
        """
        # Validate descriptor name
        valid_descriptors = self.HISTOGRAM_DESCRIPTORS
        if descriptor_name not in valid_descriptors:
            raise ValueError(f"Invalid descriptor name. Must be one of {valid_descriptors}")
        
        # Try precomputed first
        pre_val = self._lookup_precomputed(descriptor_name, is_global=False)
        if pre_val is not None:
            return pre_val

        # Get histograms for both shapes
        hist_a, bins_a = self._get_histogram(self.shape_a, descriptor_name)
        hist_b, bins_b = self._get_histogram(self.shape_b, descriptor_name)
        
        # Validate that histograms exist
        if hist_a is None or hist_b is None:
            raise ValueError(f"Histogram for descriptor '{descriptor_name}' is missing in one or both shapes")
        
        # Validate that histograms have the same length
        if len(hist_a) != len(hist_b):
            raise ValueError(f"Histograms must have the same length. Got {len(hist_a)} and {len(hist_b)}")
        
        # Compute EMD distance
        return self._emd_distance(hist_a, hist_b, bins_a)
    
    def _get_histogram(self, shape, descriptor_name):
        """
        Get histogram and bins for a specific descriptor from a shape.
        
        Args:
            shape (Shape): Shape object
            descriptor_name (str): Name of the descriptor
            
        Returns:
            tuple: (histogram, bins) as numpy arrays
        """
        hist_attr = f"{descriptor_name}_hist"
        bins_attr = f"{descriptor_name}_bins"
        
        hist = getattr(shape, hist_attr, None)
        bins = getattr(shape, bins_attr, None)
        
        return hist, bins
    
    def _emd_distance(self, hist_a, hist_b, bins):
        """
        Earth Mover's Distance (Wasserstein-1 distance) for 1D distributions.
        
        For 1D distributions, EMD can be computed as the L1 distance between 
        cumulative distribution functions (CDFs), weighted by bin widths.
        
        A CDF is the cumulative sum of the histogram - it tells you what 
        proportion of the distribution falls at or below each bin.
        
        Args:
            hist_a: First frequency histogram (already normalized)
            hist_b: Second frequency histogram (already normalized)
            bins: Bin edges (used to compute bin widths)
        
        Returns:
            float: EMD between the two histograms
        """
        # Compute cumulative distributions
        cdf_a = np.cumsum(hist_a)
        cdf_b = np.cumsum(hist_b)
        
        # Compute bin widths (distance between consecutive bin edges)
        bin_widths = np.diff(bins)
        
        # EMD is the weighted L1 distance between CDFs
        # Multiply each CDF difference by the corresponding bin width
        return np.sum(np.abs(cdf_a - cdf_b) * bin_widths)
    
    # -----------------------------
    # Global descriptor distances
    # -----------------------------
    def global_distance(self, descriptor_name: str) -> float:
        """
        Absolute difference of a global (scalar) descriptor between two shapes.
        """
        if descriptor_name not in self.GLOBAL_DESCRIPTORS:
            raise ValueError(f"Invalid global descriptor '{descriptor_name}'. Must be one of {self.GLOBAL_DESCRIPTORS}")
        
        # Try precomputed first
        pre_val = self._lookup_precomputed(descriptor_name, is_global=True)
        if pre_val is not None:
            return pre_val
        
        val_a = getattr(self.shape_a, descriptor_name, None)
        val_b = getattr(self.shape_b, descriptor_name, None)
        
        if val_a is None or val_b is None:
            raise ValueError(f"Global descriptor '{descriptor_name}' is missing in one or both shapes")
        if np.isnan(val_a) or np.isnan(val_b):
            raise ValueError(f"Global descriptor '{descriptor_name}' has NaN in one or both shapes")
        
        return float(abs(val_a - val_b))
    
    # -----------------------------
    # Unified descriptor distance
    # -----------------------------
    def descriptor_distance(self, name: str) -> float:
        """
        Compute distance for a single descriptor (histogram EMD or global absolute difference).
        """
        if name in self.HISTOGRAM_DESCRIPTORS:
            return self.histogram_distance(name)
        elif name in self.GLOBAL_DESCRIPTORS:
            return self.global_distance(name)
        else:
            raise ValueError(f"Unknown descriptor '{name}'. Not in histogram or global sets.")
    
    def all_descriptor_distances(self) -> dict:
        """
        Compute distances for all known descriptors and return as a dict.
        Keys are descriptor names, values are floats. Uses descriptor_distance
        for centralized routing.
        """
        distances = {}
        for name in self.HISTOGRAM_DESCRIPTORS + self.GLOBAL_DESCRIPTORS:
            try:
                distances[name] = self.descriptor_distance(name)
            except Exception:
                # Skip missing/invalid descriptors
                continue
        return distances
    
    # -----------------------------
    # Weighted total distance
    # -----------------------------
    def total_distance(self, weights_csv: str = "distance_weights.csv", normalize_missing: bool = True) -> float:
        """
        Compute weighted sum of descriptor distances using weights from a CSV.
        
        CSV format (created by default):
            descriptor,type,weight
            A3,histogram,0.090909
            ...
            eccentricity,global,0.090909
        
        Args:
            weights_csv: Path to CSV with columns [descriptor, weight] or [descriptor, type, weight].
                         If relative, resolved next to this file.
            normalize_missing: If True, renormalize the weights over the subset of descriptors that
                                successfully produce a distance (avoids penalizing missing data).
        Returns:
            float: Weighted total distance
        """
        if self.debug:
            import time
            t_start = time.time()
            print("[DEBUG] total_distance() called")
        
        # Resolve weights path
        if not os.path.isabs(weights_csv):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            weights_csv = os.path.join(script_dir, weights_csv)
        
        if self.debug:
            t0 = time.time()
        
        # Load distances for all descriptors
        distances = self.all_descriptor_distances()
        if not distances:
            raise ValueError("No descriptor distances could be computed for these shapes.")
        
        if self.debug:
            t1 = time.time()
            print(f"[DEBUG] all_descriptor_distances() took {(t1-t0)*1000:.2f}ms")
            print(f"[DEBUG] Got {len(distances)} distances: {list(distances.keys())}")
        
        # Load weights file
        if not os.path.exists(weights_csv):
            raise FileNotFoundError(f"Weights file not found: {weights_csv}")
        wdf = pd.read_csv(weights_csv)
        # Support both ['descriptor','weight'] and ['descriptor','type','weight']
        if 'descriptor' not in wdf.columns or 'weight' not in wdf.columns:
            raise ValueError("Weights CSV must have at least columns: 'descriptor' and 'weight'")
        
        weights = {row['descriptor']: float(row['weight']) for _, row in wdf.iterrows()}
        
        # Combine only over descriptors we have distances for and that have a weight
        used = {name: distances[name] for name in distances.keys() if name in weights}
        if not used:
            raise ValueError("No overlap between available descriptor distances and weights file.")
        
        # Determine weight normalization factor
        weight_sum = sum(weights[name] for name in used.keys())
        if weight_sum <= 0:
            raise ValueError("Sum of selected weights must be positive")
        
        if normalize_missing:
            # Renormalize so that only used descriptors sum to 1
            norm_factor = weight_sum
        else:
            # Keep original scale (sum to <= 1 if some missing)
            norm_factor = 1.0
            # But if the full weights sum to something other than 1, respect original CSV scale
            # We won't adjust further here.
        
        total = 0.0
        for name, dist in used.items():
            w = weights[name]
            w_eff = w / norm_factor if normalize_missing else w
            total += w_eff * dist
        
        if self.debug:
            t_end = time.time()
            print(f"[DEBUG] total_distance() result: {total:.6f}")
            print(f"[DEBUG] total_distance() total time: {(t_end-t_start)*1000:.2f}ms\n")
        
        return float(total)
    
    def __repr__(self):
        """String representation of ShapeDistance object."""
        return (f"ShapeDistance(shape_a='{self.shape_a.shape}', "
                f"shape_b='{self.shape_b.shape}')")


# Example usage
if __name__ == "__main__":
    # Example: Compare two shapes
    shape1 = Shape("m1337_06_fill_holes_and_orientation.obj")
    shape2 = Shape("m1338_06_fill_holes_and_orientation.obj")
    
    distance_calculator = ShapeDistance(shape1, shape2)
    
    print(f"Comparing: {shape1.shape} vs {shape2.shape}")
    print(f"Classes: {shape1.shape_class} vs {shape2.shape_class}\n")
    
    # Test different descriptors using EMD
    descriptors = ['A3', 'D1', 'D2', 'D3', 'D4']
    
    print("EMD distances for each descriptor:")
    for descriptor in descriptors:
        try:
            dist = distance_calculator.histogram_distance(descriptor)
            print(f"  {descriptor}: {dist:.6f}")
        except Exception as e:
            print(f"  {descriptor}: Error - {e}")

    # Example: Compute total weighted distance
    try:
        total = distance_calculator.total_distance()
        print(f"\nTotal weighted distance: {total:.6f}")
    except Exception as e:
        print(f"\nTotal distance error: {e}")
