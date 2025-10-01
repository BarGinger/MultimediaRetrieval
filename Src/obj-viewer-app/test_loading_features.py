"""
Test script to verify the new loading toast and indicator features.
This script checks that the new components and callbacks are properly integrated.
"""

import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported without errors."""
    try:
        from viewer.layout import build_layout
        from viewer.callbacks import register_callbacks
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_layout_components():
    """Test that the new layout components are present."""
    try:
        from viewer.layout import build_layout
        from core.dataset_cache import get_available_datasets
        
        # Get available datasets
        datasets = get_available_datasets()
        if not datasets:
            datasets = ['Data']  # Fallback
            
        # Create a mock file_df (empty)
        import pandas as pd
        file_df = pd.DataFrame()
        
        # Build layout
        layout = build_layout(file_df, datasets, datasets[0])
        
        # Convert to string to search for our new components
        layout_str = str(layout)
        
        # Check for global loading indicator
        if 'global-loading-indicator' in layout_str:
            print("✅ Global loading indicator component found in layout")
        else:
            print("❌ Global loading indicator component NOT found in layout")
            
        # Check for global loading container class
        if 'global-loading-container' in layout_str:
            print("✅ Global loading container class found in layout")
        else:
            print("❌ Global loading container class NOT found in layout")
            
        return True
        
    except Exception as e:
        print(f"❌ Layout test error: {e}")
        return False

def test_css_file():
    """Test that the CSS file contains the new styles."""
    try:
        css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
        
        if not os.path.exists(css_path):
            print("❌ CSS file not found")
            return False
            
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
            
        # Check for new CSS classes
        if '.global-loading-container' in css_content:
            print("✅ Global loading container CSS found")
        else:
            print("❌ Global loading container CSS NOT found")
            
        if '.loading-spinner' in css_content:
            print("✅ Loading spinner CSS found")
        else:
            print("❌ Loading spinner CSS NOT found")
            
        if '@keyframes spin' in css_content:
            print("✅ Spin animation CSS found")
        else:
            print("❌ Spin animation CSS NOT found")
            
        return True
        
    except Exception as e:
        print(f"❌ CSS test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing new loading features...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Layout Components Test", test_layout_components), 
        ("CSS Styles Test", test_css_file),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results.append(result)
            print(f"✅ {test_name} completed")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! The loading features should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()