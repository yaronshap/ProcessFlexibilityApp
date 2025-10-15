# Process Flexibility Simulator

An interactive Streamlit application for exploring manufacturing process flexibility based on the principles from Jordan and Graves (1995) paper "Principles on the Benefits of Manufacturing Process Flexibility".

## 🌐 Try It Online

**Live Demo:** [https://flexibility.streamlit.app/](https://flexibility.streamlit.app/)

## Features

- **Interactive Bipartite Graph**: Visualize product-plant connections with clickable edges
- **Flexibility Configuration**: Toggle connections between products and plants
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
- Set the number of plants and products using the sidebar controls
- Configure plant capacity
- Choose demand distribution type (Normal, Uniform, or Exponential)

### Network Visualization
- The bipartite graph shows products on the left and plants on the right
- Dashed gray lines represent potential connections
- Solid blue lines represent active connections (flexibility)
- Use the sidebar buttons to toggle specific connections
- Use "Full Flexibility" and "No Flexibility" buttons for quick setup
- Try "2-Flexibility" and "3-Flexibility" for advanced chain patterns

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

## Key Concepts

### Process Flexibility
Process flexibility refers to the ability of manufacturing plants to produce multiple products. This simulator helps explore:

- **Risk Pooling**: How flexibility reduces demand variability
- **Demand Smoothing**: How connections help balance production loads
- **Capacity Utilization**: How flexibility improves overall system efficiency

### Jordan & Graves Principles
The app implements key concepts from the seminal paper:
- Benefits of manufacturing flexibility
- Optimal flexibility configurations
- Trade-offs between flexibility and complexity

## Technical Details

- Built with Streamlit for interactive web interface
- Uses NetworkX for graph operations
- Plotly for interactive visualizations
- NumPy and Pandas for data analysis
- OpenPyXL for Excel file generation
- SciPy for optimization algorithms
- Real-time progress tracking with Streamlit progress bars

## Educational Use

This simulator is designed for educational purposes to help students understand:
- Manufacturing flexibility concepts
- Network optimization principles
- Demand-supply matching
- Risk management in operations

## Recent Enhancements

- ✅ **Excel Export Functionality**: Comprehensive multi-sheet Excel reports
- ✅ **Progress Tracking**: Real-time progress bars for simulations and file generation
- ✅ **Advanced Flexibility Patterns**: 2-Flexibility and 3-Flexibility chain configurations
- ✅ **Integer Demand Values**: More realistic demand simulation
- ✅ **Timestamped Downloads**: Automatic file naming with timestamps and configuration details
- ✅ **Simplified Access**: No password protection - all features immediately accessible

## Future Enhancements

- Multi-period demand scenarios
- Cost-benefit analysis
- Additional flexibility pattern options
- Interactive sensitivity analysis
