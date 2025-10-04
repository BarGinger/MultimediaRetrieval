#!/usr/bin/env python3
"""
Test script to generate analysis CSV files for both original and processed datasets
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the UnifiedPreprocessingProcessor
from normeliztion.normalize_database import UnifiedPreprocessingProcessor

def test_analysis_generation():
    """Test the enhanced analysis CSV generation"""
    print("🧪 TESTING ANALYSIS CSV GENERATION")
    print("=" * 50)
    
    try:
        # Initialize processor
        processor = UnifiedPreprocessingProcessor(target_vertices=7500)
        
        # Generate analysis for all datasets
        generated_csvs = processor.generate_analysis_for_all_datasets()
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Generated {len(generated_csvs)} CSV files")
        
        return generated_csvs
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_analysis_generation()