# Process Flexibility Simulator

An interactive Streamlit application for exploring manufacturing process flexibility based on the principles from Jordan and Graves (1995) paper "Principles on the Benefits of Manufacturing Process Flexibility".

## Features

- **Interactive Bipartite Graph**: Visualize product-plant connections with clickable edges
- **Flexibility Configuration**: Toggle connections between products and plants
- **Demand Simulation**: Generate and visualize demand distributions
- **Optimization Analysis**: Calculate flexibility benefits and system metrics
- **Real-time Metrics**: Track flexibility ratios, capacity utilization, and risk pooling benefits

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (optional, for secure deployment):
```bash
# Copy the example file
cp env_example.txt .env

# Edit .env and set your admin password
FLEXIBILITY_ADMIN_PASSWORD=your_secure_password_here
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Security Configuration

For secure deployment (especially on GitHub or public platforms):

1. **Environment Variables**: Set the `FLEXIBILITY_ADMIN_PASSWORD` environment variable
2. **Git Ignore**: Add `.env` to your `.gitignore` file to prevent committing secrets
3. **Default Password**: If no environment variable is set, the app uses a default password
4. **Advanced Options**: The 2-Flexibility and 3-Flexibility features are password-protected

### Deployment Options

**Local Development:**
- Uses default password: `flexibility2025`
- No additional setup required

**Production/Public Deployment:**
- Set environment variable: `FLEXIBILITY_ADMIN_PASSWORD=your_secure_password`
- Never commit the actual password to version control

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

### Demand Simulation
- Click "Generate Demand Data" to create demand scenarios
- View demand distribution histograms
- Analyze demand statistics and system utilization

### Optimization Analysis
- Run optimization to calculate flexibility benefits
- View risk pooling, demand smoothing, and capacity utilization metrics
- Compare different flexibility configurations

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

## Educational Use

This simulator is designed for educational purposes to help students understand:
- Manufacturing flexibility concepts
- Network optimization principles
- Demand-supply matching
- Risk management in operations

## Future Enhancements

- Advanced optimization algorithms
- Multi-period demand scenarios
- Cost-benefit analysis
- Export functionality for results
