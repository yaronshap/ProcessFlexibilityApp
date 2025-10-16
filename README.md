# Process Flexibility Simulator

An interactive Streamlit application for exploring manufacturing process flexibility based on the principles from Jordan and Graves (1995) paper "Principles on the Benefits of Manufacturing Process Flexibility".

## 🌐 Try It Online

**Live Demo:** [https://flexibility.streamlit.app/](https://flexibility.streamlit.app/)

## Features

- **Interactive Bipartite Graph**: Visualize product-factory connections with clickable edges
- **Flexibility Configuration**: Toggle connections between products and factories using compact button grid
- **Advanced Flexibility Patterns**: 2-Flexibility and 3-Flexibility chain patterns
- **Demand Simulation**: Generate and visualize demand distributions with progress tracking
- **Monte Carlo Simulation**: Run multiple replications with real-time progress bars
- **Excel Export**: Download comprehensive results with network layout, simulation data, and summary statistics
- **Real-time Metrics**: Track flexibility ratios, capacity utilization, and risk pooling benefits

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

That's it! The app will open in your browser and you can start exploring process flexibility immediately.

## Usage

### Configuration
- Set the number of factories and products using the sidebar controls
- Configure factory capacity
- Adjust truncated normal demand distribution parameters (mean, standard deviation, bounds)

### Network Visualization
- The bipartite graph shows products on the left and factories on the right
- Dashed gray lines represent potential connections
- Solid blue lines represent active connections (flexibility)
- Use the compact button grid in the sidebar to toggle specific connections
- Use "Full Flexibility" and "No Flexibility" buttons for quick setup
- Try "2-Flexibility" and "3-Flexibility" for advanced chain patterns
- Connection grid uses compact P1, P2, F1, F2 labels for efficient space usage

### Simulation & Analysis
- Click "🚀 Run Simulator" to start Monte Carlo simulation
- Watch real-time progress bars as replications complete
- View comprehensive metrics including fill rates, units sold, and units lost
- Generate Excel reports with network layout, simulation results, and summary statistics
- Download timestamped Excel files with detailed analysis

### Results Export
- Click "📊 Generate Excel File" to create comprehensive reports
- Excel files include three sheets: Network Layout, Simulation Results, and Summary Statistics
- Files are automatically timestamped and include network configuration details
- Progress tracking shows Excel generation status
- Filenames include product count, factory count, edge count, and replication count

## Technical Details

- Built with Streamlit for interactive web interface
- Uses NetworkX for graph operations
- Plotly for interactive visualizations
- NumPy and Pandas for data analysis
- OpenPyXL for Excel file generation
- SciPy for optimization algorithms
- Real-time progress tracking with Streamlit progress bars
- Custom CSS for optimized button sizing and layout
