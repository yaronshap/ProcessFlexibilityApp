# =============================================================================
# ⚠️  WARNING: THIS PROJECT IS CURRENTLY UNDER CONSTRUCTION   ⚠️
# =============================================================================
# This application is in active development. Features may be incomplete,
# unstable, or subject to change. Please use with caution.
# =============================================================================

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random
import json
import io
import os
from scipy.optimize import linprog

# Set page config
st.set_page_config(
    page_title="Process Flexibility Simulator",
    page_icon="🏭",
    layout="wide"
)

# Display construction warning banner
st.error("⚠️ **WARNING: This project is currently under construction!** ⚠️\n\n"
         "This application is in active development. Features may be incomplete, "
         "unstable, or subject to change. Please use with caution.", icon="🚧")

# Initialize session state
if 'connections' not in st.session_state:
    st.session_state.connections = {}
if 'demand_data' not in st.session_state:
    st.session_state.demand_data = {}

def create_bipartite_graph(num_plants: int, num_products: int, connections: Dict[Tuple[int, int], bool]) -> go.Figure:
    """Create an interactive bipartite graph visualization"""
    
    # Calculate positions
    pos = {}
    
    # Position products on the left (sorted with smaller numbers at top)
    for i in range(num_products):
        pos[f"Product_{i}"] = (0, num_products - 1 - i)  # Reverse order: 1 at top
    
    # Position plants on the right (sorted with smaller numbers at top)
    for i in range(num_plants):
        pos[f"Plant_{i}"] = (2, num_plants - 1 - i)  # Reverse order: 1 at top
    
    # Create traces for each edge individually to enable clicking
    traces = []
    
    # Add all possible connections
    for product_idx in range(num_products):
        for plant_idx in range(num_plants):
            is_connected = connections.get((product_idx, plant_idx), False)
            
            # Get positions
            product_pos = pos[f"Product_{product_idx}"]
            plant_pos = pos[f"Plant_{plant_idx}"]
            
            # Create individual edge trace
            edge_trace = go.Scatter(
                x=[product_pos[0], plant_pos[0]], 
                y=[product_pos[1], plant_pos[1]],
                mode='lines',
                line=dict(
                    width=3 if is_connected else 1,
                    color='blue' if is_connected else 'lightgray',
                    dash='solid' if is_connected else 'dash'
                ),
                hoverinfo='text',
                hovertext=f"Product {product_idx+1} ↔ Plant {plant_idx+1}",
                name=f"Edge_{product_idx}_{plant_idx}",
                showlegend=False,
                customdata=[(product_idx, plant_idx)],
                hovertemplate='%{hovertext}<br>Click to toggle<extra></extra>'
            )
            traces.append(edge_trace)
    
    # Create node traces (maintain the sorted order)
    product_x = [pos[f"Product_{i}"][0] for i in range(num_products)]
    product_y = [pos[f"Product_{i}"][1] for i in range(num_products)]
    
    plant_x = [pos[f"Plant_{i}"][0] for i in range(num_plants)]
    plant_y = [pos[f"Plant_{i}"][1] for i in range(num_plants)]
    
    # Product nodes
    product_trace = go.Scatter(
        x=product_x, y=product_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f"Product {i+1}" for i in range(num_products)],
        textposition="middle center",
        marker=dict(size=60, color='lightblue', line=dict(width=3, color='darkblue')),
        name='Products',
        hovertemplate='Product %{text}<extra></extra>',
        showlegend=True
    )
    traces.append(product_trace)
    
    # Plant nodes
    plant_trace = go.Scatter(
        x=plant_x, y=plant_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f"Plant {i+1}" for i in range(num_plants)],
        textposition="middle center",
        marker=dict(size=60, color='lightgreen', line=dict(width=3, color='darkgreen')),
        name='Plants',
        hovertemplate='Plant %{text}<extra></extra>',
        showlegend=True
    )
    traces.append(plant_trace)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Process Flexibility Network",
            font=dict(size=16)
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 2.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, max(num_products, num_plants)-0.5]),
        plot_bgcolor='white',
        width=800,
        height=600
    )
    
    return fig

def generate_demand_data(num_products: int, distribution_type: str, params: Dict) -> Dict[int, List[float]]:
    """Generate demand data based on distribution type"""
    demand_data = {}
    
    if distribution_type == "Truncated Normal":
        # Get parameters from the function call
        mean = params.get('mean', 100)
        std = params.get('std', 40)
        min_val = params.get('min', 20)
        max_val = params.get('max', 180)
        
        for i in range(num_products):
            # Generate normal distribution and truncate
            raw_demands = np.random.normal(mean, std, 2000)  # Generate more to account for truncation
            # Truncate to specified range
            truncated_demands = np.clip(raw_demands, min_val, max_val)
            # Take first 1000 samples
            demand_data[i] = truncated_demands[:1000]
    
    return demand_data

def calculate_flexibility_metrics(connections: Dict[Tuple[int, int], bool], 
                                num_products: int, num_plants: int) -> Dict:
    """Calculate flexibility metrics"""
    
    # Count total connections
    total_connections = sum(connections.values())
    max_possible = num_products * num_plants
    
    # Calculate flexibility ratio
    flexibility_ratio = total_connections / max_possible if max_possible > 0 else 0
    
    # Calculate average connections per product
    product_connections = {}
    for (product_idx, plant_idx), connected in connections.items():
        if connected:
            product_connections[product_idx] = product_connections.get(product_idx, 0) + 1
    
    avg_connections_per_product = np.mean(list(product_connections.values())) if product_connections else 0
    
    return {
        'total_connections': total_connections,
        'max_possible': max_possible,
        'flexibility_ratio': flexibility_ratio,
        'avg_connections_per_product': avg_connections_per_product
    }

def solve_maximal_matching(demands: List[float], plant_capacity: float, 
                          connections: Dict[Tuple[int, int], bool], 
                          num_products: int, num_plants: int) -> Tuple[List[float], float, float]:
    """
    Solve maximal matching problem using linear programming
    Returns: (flows, total_shipped, total_lost)
    """
    
    # Create flow variables for each connected edge
    edge_vars = []
    edge_indices = {}
    var_idx = 0
    
    for (product_idx, plant_idx), connected in connections.items():
        if connected:
            edge_indices[(product_idx, plant_idx)] = var_idx
            edge_vars.append(var_idx)
            var_idx += 1
    
    if not edge_vars:
        return [0.0] * len(edge_vars), 0.0, sum(demands)
    
    # Objective: maximize total flow
    c = [-1.0] * len(edge_vars)  # Negative because we minimize
    
    # Constraints
    A_ub = []
    b_ub = []
    
    # Demand constraints: sum of flows from each product <= demand
    for product_idx in range(num_products):
        constraint = [0.0] * len(edge_vars)
        for (p_idx, plant_idx), connected in connections.items():
            if connected and p_idx == product_idx:
                constraint[edge_indices[(p_idx, plant_idx)]] = 1.0
        A_ub.append(constraint)
        b_ub.append(demands[product_idx])
    
    # Capacity constraints: sum of flows to each plant <= capacity
    for plant_idx in range(num_plants):
        constraint = [0.0] * len(edge_vars)
        for (product_idx, p_idx), connected in connections.items():
            if connected and p_idx == plant_idx:
                constraint[edge_indices[(product_idx, p_idx)]] = 1.0
        A_ub.append(constraint)
        b_ub.append(plant_capacity)
    
    # Bounds: flows >= 0
    bounds = [(0, None)] * len(edge_vars)
    
    # Solve
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            flows = result.x
            total_shipped = sum(flows)
            total_lost = sum(demands) - total_shipped
            return flows, total_shipped, total_lost
        else:
            # If no solution, return zeros
            return [0.0] * len(edge_vars), 0.0, sum(demands)
    except:
        # Fallback: return zeros
        return [0.0] * len(edge_vars), 0.0, sum(demands)

def run_simulation(num_replications: int, num_products: int, num_plants: int, 
                  plant_capacity: float, connections: Dict[Tuple[int, int], bool],
                  demand_params: Dict) -> pd.DataFrame:
    """Run the simulation and return results as DataFrame"""
    
    results = []
    
    for rep in range(num_replications):
        # Set seed
        np.random.seed(rep)
        
        # Generate demands using configurable truncated normal
        demands = []
        mean = demand_params.get('mean', 100)
        std = demand_params.get('std', 40)
        min_val = demand_params.get('min', 20)
        max_val = demand_params.get('max', 180)
        
        for _ in range(num_products):
            raw_demand = np.random.normal(mean, std)
            truncated_demand = np.clip(raw_demand, min_val, max_val)
            demands.append(truncated_demand)
        
        # Solve maximal matching
        flows, total_shipped, total_lost = solve_maximal_matching(
            demands, plant_capacity, connections, num_products, num_plants
        )
        
        # Calculate metrics
        total_demand = sum(demands)
        total_capacity = plant_capacity * num_plants
        fill_rate = (total_shipped / total_demand * 100) if total_demand > 0 else 0
        
        # Create row data
        row = {
            'Replication': rep + 1,
            'Seed': rep
        }
        
        # Add demand columns
        for i, demand in enumerate(demands):
            row[f'Demand_Product_{i+1}'] = round(demand, 2)
        
        # Add capacity columns
        for i in range(num_plants):
            row[f'Capacity_Plant_{i+1}'] = plant_capacity
        
        # Add flow columns
        flow_idx = 0
        for (product_idx, plant_idx), connected in connections.items():
            if connected:
                row[f'Flow_Product_{product_idx+1}_Plant_{plant_idx+1}'] = round(flows[flow_idx], 2)
                flow_idx += 1
        
        # Add summary columns
        row['Total_Demand'] = round(total_demand, 2)
        row['Total_Capacity'] = total_capacity
        row['Units_Sold'] = round(total_shipped, 2)
        row['Units_Lost'] = round(total_lost, 2)
        row['Fill_Rate'] = round(fill_rate, 2)
        
        results.append(row)
    
    return pd.DataFrame(results)

# Main app
st.title("🏭 Process Flexibility Simulator")
st.markdown("Based on Jordan and Graves (1995): 'Principles on the Benefits of Manufacturing Process Flexibility'")

# Sidebar for inputs
# Use Streamlit's native info box for blue background
st.sidebar.info("**🏗️ Network Configuration**")

# Input fields
num_plants_products = st.sidebar.number_input("Number of Plants and Products", min_value=1, max_value=10, value=3)
num_plants = num_plants_products
num_products = num_plants_products
plant_capacity = st.sidebar.number_input("Plant Capacity", min_value=1, max_value=1000, value=100)

# Demand distribution configuration
st.sidebar.subheader("Demand Distribution")
st.sidebar.markdown("**Truncated Normal Distribution Parameters:**")

# Demand distribution parameters
demand_mean = st.sidebar.number_input("Mean", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
demand_std = st.sidebar.number_input("Standard Deviation", min_value=1.0, max_value=500.0, value=40.0, step=1.0)
demand_min = st.sidebar.number_input("Lower Bound", min_value=1.0, max_value=1000.0, value=20.0, step=1.0)
demand_max = st.sidebar.number_input("Upper Bound", min_value=1.0, max_value=1000.0, value=180.0, step=1.0)
distribution_type = "Truncated Normal"

# Initialize connections if not exists
if f'connections_{num_plants}_{num_products}' not in st.session_state:
    st.session_state[f'connections_{num_plants}_{num_products}'] = {}
    # Set "No Flexibility" as default only for new configurations
    connections = {}
    for product_idx in range(num_products):
        # Use modulo to cycle through plants, ensuring different assignments when possible
        plant_idx = product_idx % num_plants
        connections[(product_idx, plant_idx)] = True
    st.session_state[f'connections_{num_plants}_{num_products}'] = connections

connections = st.session_state[f'connections_{num_plants}_{num_products}']

# Add connection toggle functionality
def toggle_connection(product_idx: int, plant_idx: int):
    """Toggle connection between product and plant"""
    key = (product_idx, plant_idx)
    if key in connections:
        connections[key] = not connections[key]
    else:
        connections[key] = True
    st.session_state[f'connections_{num_plants}_{num_products}'] = connections
    st.rerun()

# Add connection grid controls
st.sidebar.subheader("Connection Grid")
st.sidebar.write("Check boxes to create connections:")

# Add legend
st.sidebar.write("💡 **Legend**: ✓ = Connected, ☐ = Not Connected")

# Create a more compact grid view
if st.sidebar.checkbox("Compact Grid View", value=True):
    # Compact grid: Products as rows, Plants as columns
    st.sidebar.write("**Products → Plants**")
    
    # Header row with plant labels
    header_cols = st.sidebar.columns(num_plants + 1)
    with header_cols[0]:
        st.write("**Product**")
    for plant_idx in range(num_plants):
        with header_cols[plant_idx + 1]:
            st.write(f"**Plant {plant_idx+1}**")
    
    # Data rows
    for product_idx in range(num_products):
        row_cols = st.sidebar.columns(num_plants + 1)
        
        # Product label
        with row_cols[0]:
            st.write(f"Product {product_idx+1}")
        
        # Checkboxes for each plant
        for plant_idx in range(num_plants):
            key = (product_idx, plant_idx)
            is_connected = connections.get(key, False)
            
            with row_cols[plant_idx + 1]:
                if st.checkbox(
                    "",
                    value=is_connected,
                    key=f"compact_{product_idx}_{plant_idx}"
                ):
                    if not is_connected:
                        connections[key] = True
                        st.session_state[f'connections_{num_plants}_{num_products}'] = connections
                        st.rerun()
                else:
                    if is_connected:
                        connections[key] = False
                        st.session_state[f'connections_{num_plants}_{num_products}'] = connections
                        st.rerun()

else:
    # Detailed view: Each product gets its own section
    for product_idx in range(num_products):
        st.sidebar.write(f"**Product {product_idx+1}**")
        
        # Create columns for plants
        cols = st.sidebar.columns(num_plants)
        
        for plant_idx, col in enumerate(cols):
            key = (product_idx, plant_idx)
            is_connected = connections.get(key, False)
            
            with col:
                if st.checkbox(
                    f"Plant {plant_idx+1}",
                    value=is_connected,
                    key=f"detailed_{product_idx}_{plant_idx}"
                ):
                    if not is_connected:
                        connections[key] = True
                        st.session_state[f'connections_{num_plants}_{num_products}'] = connections
                        st.rerun()
                else:
                    if is_connected:
                        connections[key] = False
                        st.session_state[f'connections_{num_plants}_{num_products}'] = connections
                        st.rerun()
        
        st.sidebar.write("---")  # Separator between products

# Quick setup buttons
st.sidebar.subheader("Quick Setup")

# First row: Main flexibility options
col_a, col_b, col_c = st.sidebar.columns(3)

with col_a:
    if st.button("Full Flexibility", help="Connect all products to all plants"):
        for product_idx in range(num_products):
            for plant_idx in range(num_plants):
                connections[(product_idx, plant_idx)] = True
        st.session_state[f'connections_{num_plants}_{num_products}'] = connections
        # Clear previous simulation results since network configuration changed
        if 'simulation_completed' in st.session_state:
            del st.session_state.simulation_completed
        if 'simulation_results' in st.session_state:
            del st.session_state.simulation_results
        st.rerun()

with col_b:
    if st.button("No Flexibility", help="Connect each product to one plant (dedicated assignment)"):
        connections.clear()
        # Connect each product to a single plant, preferably different ones
        for product_idx in range(num_products):
            # Use modulo to cycle through plants, ensuring different assignments when possible
            plant_idx = product_idx % num_plants
            connections[(product_idx, plant_idx)] = True
        st.session_state[f'connections_{num_plants}_{num_products}'] = connections
        # Clear previous simulation results since network configuration changed
        if 'simulation_completed' in st.session_state:
            del st.session_state.simulation_completed
        if 'simulation_results' in st.session_state:
            del st.session_state.simulation_results
        st.rerun()

with col_c:
    if st.button("Remove Connections", help="Remove all connections"):
        connections.clear()
        st.session_state[f'connections_{num_plants}_{num_products}'] = connections
        # Clear previous simulation results since network configuration changed
        if 'simulation_completed' in st.session_state:
            del st.session_state.simulation_completed
        if 'simulation_results' in st.session_state:
            del st.session_state.simulation_results
        st.rerun()





# Main content area - just the graph
# Create the graph
fig = create_bipartite_graph(num_plants, num_products, connections)
st.plotly_chart(fig, use_container_width=True)

# Instructions
st.info("💡 **Instructions**: Use the checkboxes in the sidebar to toggle connections between products and plants. Solid blue lines indicate active connections (flexibility).")


# Add horizontal divider
st.sidebar.markdown("---")

# Use Streamlit's native success box for green background
st.sidebar.success("**📊 Analysis & Simulation**")

# Replication input and simulator button
st.sidebar.subheader("Simulation Parameters")
num_replications = st.sidebar.number_input("Number of Replications", min_value=1, max_value=10000, value=100, step=100)

if st.sidebar.button("🚀 Run Simulator", type="primary"):
    with st.spinner(f"Running simulation with {num_replications} replications..."):
        # Run the simulation
        demand_params = {
            'mean': demand_mean,
            'std': demand_std,
            'min': demand_min,
            'max': demand_max
        }
        simulation_results = run_simulation(
            num_replications, num_products, num_plants, plant_capacity, connections, demand_params
        )
        
        # Store results in session state
        st.session_state.simulation_results = simulation_results
        st.session_state.simulation_completed = True
    
    st.sidebar.success(f"Simulation completed with {num_replications} replications!")

# Display simulation results and download button
if st.session_state.get('simulation_completed', False):
    st.subheader("Simulation Results")
    
    # Display summary statistics
    results_df = st.session_state.simulation_results
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Fill Rate", f"{results_df['Fill_Rate'].mean():.1f}%")
    with col2:
        avg_sold = results_df['Units_Sold'].mean()
        std_sold = results_df['Units_Sold'].std()
        st.metric("Average Units Sold", f"{avg_sold:.1f} ± {std_sold:.1f}")
    with col3:
        avg_lost = results_df['Units_Lost'].mean()
        std_lost = results_df['Units_Lost'].std()
        st.metric("Average Units Lost", f"{avg_lost:.1f} ± {std_lost:.1f}")
    with col4:
        st.metric("Total Replications", len(results_df))
    
    # Display first few rows
    st.subheader("Sample Results (First 10 Replications)")
    st.dataframe(results_df.head(10), use_container_width=True)
    
    # Download button
    csv_data = results_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV Results",
        data=csv_data,
        file_name=f"flexibility_simulation_{num_products}products_{num_plants}plants_{num_replications}reps.csv",
        mime="text/csv",
        type="primary"
    )

# Display demand data if available
if st.session_state.demand_data:
    st.subheader("Demand Distribution Visualization")
    
    # Create demand plots
    fig_demand = go.Figure()
    
    for product_idx, demands in st.session_state.demand_data.items():
        fig_demand.add_trace(go.Histogram(
            x=demands,
            name=f'Product {product_idx}',
            opacity=0.7
        ))
    
    fig_demand.update_layout(
        title="Demand Distribution by Product",
        xaxis_title="Demand",
        yaxis_title="Frequency",
        barmode='overlay'
    )
    
    st.plotly_chart(fig_demand, use_container_width=True)
    
    # Display summary statistics
    st.subheader("Demand Summary Statistics")
    summary_data = []
    for product_idx, demands in st.session_state.demand_data.items():
        summary_data.append({
            'Product': f'Product {product_idx}',
            'Mean': np.mean(demands),
            'Std': np.std(demands),
            'Min': np.min(demands),
            'Max': np.max(demands)
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)
    
    # Optimization results
    st.subheader("Optimization Results")
    
    # Calculate theoretical benefits
    total_demand = sum([np.mean(demands) for demands in st.session_state.demand_data.values()])
    total_capacity = plant_capacity * num_plants
    
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    
    with col_opt1:
        st.metric("Total Demand", f"{total_demand:.0f}")
    
    with col_opt2:
        st.metric("Total Capacity", f"{total_capacity:.0f}")
    
    with col_opt3:
        utilization = (total_demand / total_capacity) * 100 if total_capacity > 0 else 0
        st.metric("System Utilization", f"{utilization:.1f}%")
    
    # Flexibility benefits analysis
    st.subheader("Flexibility Benefits Analysis")
    
    # Calculate flexibility benefits based on Jordan & Graves principles
    flexibility_benefits = {
        'Risk Pooling': f"{(metrics['flexibility_ratio'] * 100):.1f}% risk reduction",
        'Demand Smoothing': f"{(1 - metrics['flexibility_ratio']) * 50:.1f}% demand variability reduction",
        'Capacity Utilization': f"{min(100, utilization + (metrics['flexibility_ratio'] * 20)):.1f}%"
    }
    
    for benefit, value in flexibility_benefits.items():
        st.metric(benefit, value)

# Credits section at the bottom of sidebar
st.sidebar.markdown("---")
st.sidebar.warning("**📝 Credits**")

st.sidebar.markdown("**Created by:** Yaron Shaposhnik")
st.sidebar.markdown("**Email:** yaron.shaposhnik@simon.rochester.edu")
st.sidebar.markdown("**Year:** 2025")
st.sidebar.markdown("*Process Flexibility Simulation Tool*")

# Advanced options button at the bottom
if st.sidebar.button("Advanced Options", key="advanced_options", help="Click to show advanced options"):
    st.session_state.show_advanced = True

# Password protection for advanced options (appears right below unlock button)
if st.session_state.get('show_advanced', False):
    password = st.sidebar.text_input("Enter password for advanced options:", type="password")
    # Get password from environment variable or use default
    admin_password = os.getenv("FLEXIBILITY_ADMIN_PASSWORD", "flexibility2025")
    if password == admin_password:
        st.session_state.advanced_unlocked = True
        st.sidebar.success("Advanced options unlocked!")
    elif password != "":
        st.sidebar.error("Incorrect password")
    
    if st.sidebar.button("🔒 Hide Advanced Options"):
        st.session_state.show_advanced = False
        st.session_state.advanced_unlocked = False
        st.rerun()

# Advanced flexibility buttons (only show if unlocked)
if st.session_state.get('advanced_unlocked', False):
    st.sidebar.markdown("**Advanced Flexibility Options:**")
    col_d, col_e = st.sidebar.columns(2)
    
    with col_d:
        if st.sidebar.button("2-Flexibility", help="Create 2-chain flexibility pattern"):
            connections.clear()
            # Create 2-chain pattern: P1→Plant1,Plant2; P2→Plant2,Plant3; P3→Plant3,Plant1; etc.
            for product_idx in range(num_products):
                # Each product connects to 2 plants in a chain pattern
                plant1_idx = product_idx % num_plants
                plant2_idx = (product_idx + 1) % num_plants
                connections[(product_idx, plant1_idx)] = True
                connections[(product_idx, plant2_idx)] = True
            st.session_state[f'connections_{num_plants}_{num_products}'] = connections
            # Clear previous simulation results since network configuration changed
            if 'simulation_completed' in st.session_state:
                del st.session_state.simulation_completed
            if 'simulation_results' in st.session_state:
                del st.session_state.simulation_results
            st.rerun()

    with col_e:
        if st.sidebar.button("3-Flexibility", help="Create 3-chain flexibility pattern"):
            connections.clear()
            # Create 3-chain pattern: P1→Plant1,Plant2,Plant3; P2→Plant2,Plant3,Plant1; etc.
            for product_idx in range(num_products):
                # Each product connects to 3 plants in a chain pattern
                plant1_idx = product_idx % num_plants
                plant2_idx = (product_idx + 1) % num_plants
                plant3_idx = (product_idx + 2) % num_plants
                connections[(product_idx, plant1_idx)] = True
                connections[(product_idx, plant2_idx)] = True
                connections[(product_idx, plant3_idx)] = True
            st.session_state[f'connections_{num_plants}_{num_products}'] = connections
            # Clear previous simulation results since network configuration changed
            if 'simulation_completed' in st.session_state:
                del st.session_state.simulation_completed
            if 'simulation_results' in st.session_state:
                del st.session_state.simulation_results
            st.rerun()

