#!/usr/bin/env python3
"""
Test script to check for import errors and basic functionality
"""

try:
    print("Testing imports...")
    import streamlit as st
    print("✓ Streamlit imported successfully")
    
    import networkx as nx
    print("✓ NetworkX imported successfully")
    
    import plotly.graph_objects as go
    print("✓ Plotly imported successfully")
    
    import numpy as np
    print("✓ NumPy imported successfully")
    
    import pandas as pd
    print("✓ Pandas imported successfully")
    
    print("\nAll imports successful! The issue might be elsewhere.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing packages with: pip install -r requirements.txt")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("\nTesting basic Streamlit functionality...")
try:
    # Test basic streamlit functionality
    st.set_page_config(page_title="Test")
    print("✓ Streamlit basic functionality works")
except Exception as e:
    print(f"❌ Streamlit error: {e}")
