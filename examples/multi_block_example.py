"""
EXAMPLE: Multi-Block Mesh Visualization

This example demonstrates how to work with Exodus meshes that contain multiple
element blocks. Multi-block meshes are common in simulations with:
- Multiple materials (e.g., different phases or components)
- Domain decomposition (different regions with different properties)
- Adaptive mesh refinement (different refinement levels)

The block_id parameter allows you to extract data from specific blocks,
enabling block-by-block analysis and visualization.
"""

from ExodusReader import ExodusReader
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from plotting_utils import setup_publication_style, get_scientific_colormap, save_figure

# Setup publication-quality plotting
setup_publication_style(use_latex=False)

# Get scientific colormap
cmap = get_scientific_colormap('batlow')

# EXAMPLE EXODUS FILE WITH MULTIPLE BLOCKS
# Replace this with your actual multi-block mesh file
filenames = '2D/multi_block_mesh.e*'

# OPEN EXODUS FILE WITH CONTEXT MANAGER
with ExodusReader(filenames) as reader:

    # STEP 1: Inspect available blocks
    print("=" * 60)
    print("MULTI-BLOCK MESH INFORMATION")
    print("=" * 60)
    print(f"Available element blocks: {list(reader.block_connect.keys())}")
    print(f"Number of blocks: {len(reader.block_connect)}")

    # Print details about each block
    for block_id in reader.block_connect.keys():
        n_elements = len(reader.block_connect[block_id])
        print(f"  Block {block_id}: {n_elements} elements")

    print("\nAvailable nodal variables:", reader.nodal_var_names)
    print("Available element variables:", reader.elem_var_names)
    print("=" * 60)

    # STEP 2: Get simulation times
    times = reader.global_times
    print(f"\nSimulation timesteps: {len(times)}")
    print(f"Time range: {times[0]:.2e} to {times[-1]:.2e}")

    # Select a timestep to visualize (last timestep)
    time = times[-1]

    # STEP 3: Choose a variable to plot (modify this to match your data)
    # Common variables: 'temperature', 'unique_grains', 'stress', 'strain', etc.
    var_name = 'temperature'  # Change this to match your simulation

    # Check if variable exists
    if var_name not in reader.elem_var_names and var_name not in reader.nodal_var_names:
        print(f"\nWARNING: Variable '{var_name}' not found in mesh.")
        print("Please edit this script to use one of the available variables listed above.")
        exit()

    # STEP 4: Extract and plot data from each block
    # Create subplots - one for each block plus one for the full mesh
    block_ids = list(reader.block_connect.keys())
    n_blocks = len(block_ids)

    # Determine subplot layout
    if n_blocks <= 2:
        nrows, ncols = 1, n_blocks + 1
        figsize = (7 * (n_blocks + 1) / 2, 3)
    else:
        nrows = 2
        ncols = (n_blocks + 2) // 2
        figsize = (7, 6)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=500)
    if n_blocks == 1:
        axes = [axes]  # Make it iterable
    else:
        axes = axes.flatten()

    # STEP 5: Plot full mesh (all blocks combined)
    print(f"\nExtracting full mesh data for variable '{var_name}'...")
    x_full, y_full, z_full, c_full = reader.get_data_at_time(var_name, time)

    # Create coordinates array for polygon plotting
    coords_full = np.asarray([np.asarray([x_val, y_val]).T for (x_val, y_val) in zip(x_full, y_full)])

    # Plot full mesh in first subplot
    ax = axes[0]
    p_full = PolyCollection(coords_full, cmap=cmap, alpha=1.0, edgecolor='k', linewidth=0.5)
    p_full.set_array(np.array(c_full))
    ax.add_collection(p_full)

    # Set axis limits based on data
    x_min, x_max = np.min(x_full), np.max(x_full)
    y_min, y_max = np.min(y_full), np.max(y_full)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal')
    ax.set_title('Full Mesh (All Blocks)', fontweight='bold')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')

    # Add colorbar for full mesh
    cbar_full = fig.colorbar(p_full, ax=ax, fraction=0.046, pad=0.04)
    cbar_full.ax.set_ylabel(var_name, fontweight='bold')

    # STEP 6: Plot each block separately
    # Track global min/max for consistent color scaling across blocks
    c_global_min = np.min(c_full)
    c_global_max = np.max(c_full)

    for idx, block_id in enumerate(block_ids):
        print(f"Processing block {block_id}...")

        # Extract data for this specific block
        x_block, y_block, z_block, c_block = reader.get_data_at_time(var_name, time, block_id=block_id)

        # Create coordinates array
        coords_block = np.asarray([np.asarray([x_val, y_val]).T for (x_val, y_val) in zip(x_block, y_block)])

        # Plot in corresponding subplot (offset by 1 since axes[0] is full mesh)
        ax = axes[idx + 1]
        p_block = PolyCollection(coords_block, cmap=cmap, alpha=1.0, edgecolor='k', linewidth=0.5)
        p_block.set_array(np.array(c_block))

        # Use global color limits for consistent comparison
        p_block.set_clim(c_global_min, c_global_max)

        ax.add_collection(p_block)

        # Set axis limits
        x_block_min, x_block_max = np.min(x_block), np.max(x_block)
        y_block_min, y_block_max = np.min(y_block), np.max(y_block)
        ax.set_xlim([x_block_min, x_block_max])
        ax.set_ylim([y_block_min, y_block_max])
        ax.set_aspect('equal')
        ax.set_title(f'Block {block_id}', fontweight='bold')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')

        # Add colorbar
        cbar_block = fig.colorbar(p_block, ax=ax, fraction=0.046, pad=0.04)
        cbar_block.ax.set_ylabel(var_name, fontweight='bold')

        # Print block statistics
        print(f"  Block {block_id}: {var_name} range = [{np.min(c_block):.3e}, {np.max(c_block):.3e}]")

    # Hide unused subplots if any
    for idx in range(n_blocks + 1, len(axes)):
        axes[idx].set_visible(False)

    print("\nRendering complete!")
    print("=" * 60)

# STEP 7: Save the figure
save_figure(fig, 'multi_block_comparison', formats=['png', 'pdf'], dpi=500, transparent=True)

# Show the plot
plt.show()

print("\n" + "=" * 60)
print("USAGE NOTES")
print("=" * 60)
print("1. This is a TEMPLATE script - you must adapt it to your specific mesh")
print("2. Update 'filenames' to point to your multi-block Exodus file")
print("3. Update 'var_name' to match a variable in your simulation")
print("4. The script will automatically detect and plot all available blocks")
print("5. Color scaling is consistent across all subplots for easy comparison")
print("=" * 60)
