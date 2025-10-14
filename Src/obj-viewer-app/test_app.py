#!/usr/bin/env python3
"""
Quick test of the step slider functionality
"""

import sys
sys.path.append('.')

from viewer.init import create_dash_app

def main():
    """Create and run a simple test of the app"""
    print("🧪 Testing Step Slider Implementation")
    print("=" * 50)
    print("Starting 3D Shape Viewer with step navigation...")
    print("🔧 Features to test:")
    print("  • UnifiedPreprocessed/Data dataset selection")
    print("  • Step slider activation for processed shapes")
    print("  • Step navigation (0-5: Original → Remeshed → Translated → Aligned → Flipped → Scaled)")
    print("  • Slider disabled for non-processed shapes")
    print("\n🌐 Open your browser to: http://127.0.0.1:8050")
    print("📋 Test checklist:")
    print("  1. Select 'UnifiedPreprocessed/Data' from dataset dropdown")
    print("  2. Click on any shape in the file list")
    print("  3. Verify step slider is enabled (shows blue gradient)")
    print("  4. Move slider from 0-5 and watch shape change")
    print("  5. Check that step info updates (e.g., 'Step 3: Aligned - PCA aligned to axes')")
    print("  6. Try original 'Data' dataset - slider should be disabled")
    print()
    
    app = create_dash_app()
    app.run(debug=True, host="127.0.0.1", port=8050)

if __name__ == "__main__":
    main()