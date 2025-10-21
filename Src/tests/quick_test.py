#!/usr/bin/env python3
"""
Quick script to start app briefly and check console output.
"""

import sys
from pathlib import Path
import time
import threading

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🚀 Starting app briefly to check console output...")
    
    try:
        # Import and setup
        from viewer.init import create_dash_app
        
        print("🔧 Creating app...")
        app = create_dash_app()
        
        # Run for a short time to see initialization logs
        def stop_app():
            time.sleep(3)  # Wait 3 seconds
            print("\n⏹️ Stopping app...")
            import os
            os._exit(0)
        
        # Start stop timer
        stop_thread = threading.Thread(target=stop_app)
        stop_thread.daemon = True
        stop_thread.start()
        
        # Start app
        print("🌐 App starting on http://127.0.0.1:8050")
        app.run(debug=False, host='127.0.0.1', port=8050)
        
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()