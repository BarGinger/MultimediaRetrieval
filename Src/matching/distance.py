import numpy as np
from shapeFeatures import Shape


class ShapeDistance:
    """
    A class for computing distance metrics between 3D shapes.
    
    Supports various distance measures for histogram-based shape descriptors
    and global shape features.
    """
    
    def __init__(self, shape_a, shape_b):
        """
        Initialize ShapeDistance with two Shape objects.
        
        Args:
            shape_a (Shape): First shape to compare
            shape_b (Shape): Second shape to compare
        """
        if not isinstance(shape_a, Shape) or not isinstance(shape_b, Shape):
            raise TypeError("Both arguments must be Shape objects")
        
        self.shape_a = shape_a
        self.shape_b = shape_b
    
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
        valid_descriptors = ['A3', 'D1', 'D2', 'D3', 'D4']
        if descriptor_name not in valid_descriptors:
            raise ValueError(f"Invalid descriptor name. Must be one of {valid_descriptors}")
        
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
