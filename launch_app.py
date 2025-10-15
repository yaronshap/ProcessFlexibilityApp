#!/usr/bin/env python3
"""
Simple launcher script for the Process Flexibility Simulator
This script will help diagnose and launch the Streamlit app
"""

import sys
import os
import subprocess

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = ['streamlit', 'networkx', 'plotly', 'numpy', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True

def launch_app():
    """Launch the Streamlit app"""
    try:
        print("Launching Process Flexibility Simulator...")
        print("The app will open in your default web browser.")
        print("Press Ctrl+C to stop the app.")
        print("-" * 50)
        
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        
    except KeyboardInterrupt:
        print("\nApp stopped by user.")
    except Exception as e:
        print(f"Error launching app: {e}")
        print("Please make sure you're in the correct directory and all dependencies are installed.")

if __name__ == "__main__":
    print("Process Flexibility Simulator Launcher")
    print("=" * 40)
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found!")
        print("Please make sure you're in the correct directory.")
        sys.exit(1)
    
    print("✓ app.py found")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install missing requirements first.")
        sys.exit(1)
    
    print("\n✓ All requirements satisfied")
    
    # Launch the app
    launch_app()
