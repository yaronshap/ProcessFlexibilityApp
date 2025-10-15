#!/usr/bin/env python3
"""
Test script to verify the Plotly titlefont fix
"""

try:
    import plotly.graph_objects as go
    import numpy as np
    
    print("Testing Plotly titlefont fix...")
    
    # Test the old (deprecated) way
    try:
        fig_old = go.Figure()
        fig_old.update_layout(
            title="Test Title",
            titlefont_size=16  # This should cause an error in newer Plotly versions
        )
        print("❌ Old method still works - this might indicate an older Plotly version")
    except Exception as e:
        print(f"✓ Old method correctly fails: {type(e).__name__}")
    
    # Test the new (correct) way
    try:
        fig_new = go.Figure()
        fig_new.update_layout(
            title=dict(
                text="Test Title",
                font=dict(size=16)
            )
        )
        print("✓ New method works correctly")
    except Exception as e:
        print(f"❌ New method failed: {e}")
    
    print("\nPlotly version test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install plotly: pip install plotly")
