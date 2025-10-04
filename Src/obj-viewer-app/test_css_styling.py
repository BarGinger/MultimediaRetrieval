#!/usr/bin/env python3
"""
Test to force missing step styling for debugging
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

def test_force_missing_styling():
    """Test if we can force the missing step styling by manually setting classes"""
    print("🧪 Testing CSS styling for missing steps...")
    
    # Check if CSS file has the right styling
    css_file = src_dir / "assets" / "style.css"
    css_content = css_file.read_text()
    
    # Look for the missing step styling
    if ".step-label.missing" in css_content:
        print("✅ Found .step-label.missing in CSS")
        
        # Extract the styling
        lines = css_content.split('\n')
        in_missing_block = False
        missing_styles = []
        
        for line in lines:
            if '.step-label.missing {' in line:
                in_missing_block = True
                missing_styles.append(line)
            elif in_missing_block:
                if line.strip() == '}':
                    missing_styles.append(line)
                    break
                missing_styles.append(line)
        
        print("📝 Missing step CSS styling:")
        for style in missing_styles:
            print(f"   {style}")
    else:
        print("❌ .step-label.missing not found in CSS")
    
    # Test HTML structure that should be generated
    print("\n🔧 Expected HTML structure:")
    print('   <div class="step-label missing" id="step-label-1">Mesh</div>')
    
    # Test if classes are being applied correctly
    print("\n🎯 Expected class names from callback:")
    class_names = [
        "step-label",           # Step 0 (available)
        "step-label missing",   # Step 1 (missing)  
        "step-label active",    # Step 2 (active)
        "step-label",           # Step 3 (available)
        "step-label",           # Step 4 (available)
        "step-label"            # Step 5 (available)
    ]
    
    for i, class_name in enumerate(class_names):
        print(f"   Step {i}: '{class_name}'")

if __name__ == "__main__":
    test_force_missing_styling()