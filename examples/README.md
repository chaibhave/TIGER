# TIGER Examples

This directory contains example scripts demonstrating how to use TIGER to visualize Exodus simulation data. All examples follow modern best practices including:
- Context manager usage for proper resource management
- Publication-quality plotting with scientific colormaps
- Consistent figure formatting and output

## Prerequisites

1. **Install TIGER**: Follow the installation instructions in the main README
2. **Activate environment**: `source tiger_env/bin/activate`
3. **Create output directories**: `mkdir -p 1D 2D 3D`
4. **Prepare data files**: Place your Exodus files in the appropriate directories

## Example Scripts

### Basic Examples

#### `1d_plot.py` - 1D Line Plot
**Purpose**: Demonstrates plotting 1D simulation data as line plots.

**Required data**: 1D Exodus file (e.g., `1D/Ni20Cr_hart.e*`)

**Features**:
- Plots data from multiple timesteps on the same figure
- Uses `block_id` parameter to extract specific element block
- Legend positioned below plot to avoid data overlap

**Usage**:
```bash
python 1d_plot.py
```

**Output**: `1D/1d_exodus_line_plot.png` and `.pdf`

---

#### `2d_plot.py` - Basic 2D Mesh Visualization
**Purpose**: Renders 2D mesh with element-based coloring for multiple timesteps.

**Required data**: 2D Exodus file (e.g., `2D/grain_growth_2D_graintracker_out.e*`)

**Features**:
- Loops through timesteps to generate animation frames
- Uses PolyCollection for efficient polygon rendering
- Demonstrates frame-by-frame rendering for video generation

**Usage**:
```bash
python 2d_plot.py
```

**Output**: Series of PNG files: `2D/2d_render_0.png`, `2D/2d_render_1.png`, ...

**Video generation** (optional, requires opencv-python):
```bash
# After generating frames, create video with ffmpeg or similar tools
ffmpeg -framerate 20 -pattern_type glob -i '2D/2d_render_*.png' -c:v libx264 -pix_fmt yuv420p 2D/animation.mp4
```

---

#### `3d_plot.py` - 3D Mesh Visualization
**Purpose**: Renders 3D volumetric meshes with element coloring.

**Required data**: 3D Exodus file (e.g., `3D/grain_growth_3D_out.e*`)

**Features**:
- 3D visualization using Poly3DCollection
- Interactive rotation and zoom with matplotlib 3D viewer
- Replaces HSV colormap with perceptually uniform batlow

**Usage**:
```bash
python 3d_plot.py
```

**Output**: `3D/3d_render.png` and `.pdf`

**Note**: 3D plotting can be slow for large meshes. Use plt.show() for interactive viewing.

---

### Advanced Examples

#### `2d_line_plot.py` - 2D Line Sampling + Visualization
**Purpose**: Extracts variable values along a line through 2D mesh and plots both the mesh and line profile.

**Required data**: 2D Exodus file

**Features**:
- Defines line with two endpoints (P1, P2)
- Samples variable values along the line
- Creates side-by-side figure: mesh with line overlay + line profile plot

**Usage**:
```bash
python 2d_line_plot.py
```

**Output**: `2d_lineplot.png` and `.pdf`

**Customization**: Edit P1, P2, and n_points in the script to change sampling line.

---

#### `2d_plot_advanced.py` - Combined Exodus + CSV Visualization
**Purpose**: Demonstrates plotting Exodus mesh alongside postprocessor CSV data (e.g., grain counts over time).

**Required data**:
- 2D Exodus file: `2D/grain_growth_2D_graintracker_out.e*`
- CSV postprocessor file: `2D/grain_growth_2D_graintracker_out.csv`

**Features**:
- Reads CSV data using the included `getRawData()` function
- Synchronizes Exodus rendering with postprocessor data timeline
- Creates two-panel figure: mesh visualization + time series plot

**Usage**:
```bash
python 2d_plot_advanced.py
```

**Output**: Series of PNG files: `2D/2d_fancy_0.png`, `2D/2d_fancy_1.png`, ...

---

#### `2d_plot_parallel.py` - Parallel Frame Rendering
**Purpose**: Uses Python multiprocessing to render frames in parallel for faster video generation.

**Required data**: 2D Exodus file

**Features**:
- Distributes rendering across multiple CPU cores
- Configurable number of processes (n_procs)
- Significantly faster than serial rendering for large frame counts

**Usage**:
```bash
python 2d_plot_parallel.py
```

**Output**: Series of PNG files: `2D/2d_render_0.png`, `2D/2d_render_1.png`, ...

**Performance**: Adjust `n_procs` based on available CPU cores. Be mindful of memory usage.

---

#### `multi_block_example.py` - Multi-Block Mesh Demonstration
**Purpose**: Template for working with meshes containing multiple element blocks (multi-material or multi-region simulations).

**Required data**: Exodus file with multiple element blocks

**Features**:
- Automatically detects and lists available blocks
- Extracts and visualizes data from each block separately
- Side-by-side comparison with consistent color scaling
- Demonstrates `block_id` parameter usage

**Usage**:
```bash
python multi_block_example.py
```

**Output**: `multi_block_comparison.png` and `.pdf`

**Important**: This is a TEMPLATE. You must edit the script to:
- Point to your actual multi-block mesh file
- Specify a variable name that exists in your simulation
- Adjust subplot layout if you have many blocks

---

## Shared Utilities

### `plotting_utils.py` - Publication-Quality Plotting Module

This module provides shared utilities used by all examples:

**Functions**:
- `setup_publication_style()`: Configures matplotlib for publication-ready figures
- `get_scientific_colormap(name)`: Returns perceptually uniform colormaps (via cmcrameri)
- `smart_legend_placement(ax)`: Auto-positions legends to avoid data overlap
- `save_figure(fig, filename)`: Saves figures in multiple formats with consistent settings

**Usage in your own scripts**:
```python
from plotting_utils import setup_publication_style, get_scientific_colormap, save_figure

setup_publication_style()
cmap = get_scientific_colormap('batlow')
# ... create your plot ...
save_figure(fig, 'my_plot')
```

**Scientific colormaps** (requires cmcrameri package):
- `batlow`: Perceptually uniform rainbow (general use)
- `vik`: Diverging colormap (for data with meaningful zero)
- `devon`: Sequential blue-to-yellow
- `lajolla`: Sequential light-to-dark

If cmcrameri is not installed, matplotlib colormaps are used as fallback.

---

## Directory Structure

```
examples/
├── README.md              # This file
├── plotting_utils.py      # Shared plotting utilities
├── 1d_plot.py            # 1D line plot example
├── 2d_plot.py            # Basic 2D rendering
├── 2d_line_plot.py       # 2D mesh + line profile
├── 2d_plot_advanced.py   # Exodus + CSV combined
├── 2d_plot_parallel.py   # Parallel frame rendering
├── 3d_plot.py            # 3D mesh visualization
├── multi_block_example.py # Multi-block demonstration
├── 1D/                   # Output directory for 1D plots
├── 2D/                   # Output directory for 2D plots
└── 3D/                   # Output directory for 3D plots
```

---

## Generating Test Data

If you don't have Exodus files, you can generate test data using MOOSE:

1. **Install MOOSE**: Follow instructions at https://mooseframework.inl.gov
2. **Run example simulations**: MOOSE includes example input files (`.i` files) that generate Exodus output
3. **Example command**:
   ```bash
   ~/projects/moose/modules/phase_field/examples/grain_growth/grain_growth_2D.i
   ~/projects/moose/modules/phase_field/phase_field-opt -i grain_growth_2D.i
   ```
4. **Move output**: Copy generated `.e` files to the appropriate TIGER directory

---

## Dependencies

**Core dependencies** (installed via requirements.txt):
- `matplotlib` - Plotting library
- `numpy` - Array operations
- `scipy` - Scientific computing
- `h5py` - HDF5 file I/O
- `netCDF4` - NetCDF/Exodus file reading
- `mpi4py` - Parallel processing support
- `pytest` - Testing framework
- `cmcrameri` - Scientific colormaps

**Optional dependencies** (requirements-optional.txt):
- `opencv-python` - Video generation from frames

---

## Common Issues and Solutions

### Issue: "No module named 'ExodusReader'"
**Solution**: Make sure you've activated the virtual environment and installed TIGER:
```bash
source tiger_env/bin/activate
pip install -e .
```

### Issue: "File not found" errors
**Solution**: Create the output directories:
```bash
mkdir -p examples/1D examples/2D examples/3D
```

### Issue: "Variable not found in mesh"
**Solution**: Check available variables in your Exodus file. You can list them using:
```python
with ExodusReader('your_file.e') as reader:
    print("Nodal variables:", reader.nodal_var_names)
    print("Element variables:", reader.elem_var_names)
```

### Issue: Slow 3D rendering
**Solution**:
- Reduce mesh resolution in your simulation
- Use a subset of timesteps for rendering
- Consider using ParaView for large 3D datasets

### Issue: cmcrameri colormaps not available
**Solution**: Install the optional package:
```bash
pip install cmcrameri
```
Or accept the fallback to matplotlib colormaps (automatic).

---

## Performance Tips

1. **For large meshes**: Use parallel rendering (`2d_plot_parallel.py`)
2. **For video generation**: Render at lower DPI (e.g., 300 instead of 500) for drafts
3. **Memory management**: Close figures with `plt.close(fig)` in loops
4. **Timestep selection**: Use `np.linspace()` to select subset of timesteps for rendering

---

## Further Reading

- **ExodusReader API**: See main repository README
- **Matplotlib documentation**: https://matplotlib.org/
- **cmcrameri colormaps**: https://www.fabiocrameri.ch/colourmaps/
- **MOOSE Framework**: https://mooseframework.inl.gov (for generating test data)
- **Exodus format specification**: https://sandialabs.github.io/seacas-docs/

---

## Contributing

If you create new examples or improve existing ones, please consider contributing back to the repository:
1. Fork the repository
2. Create a feature branch
3. Add your example with documentation
4. Submit a pull request

---

## Questions?

For questions or issues:
- Open an issue on GitHub: https://github.com/chaitanyaBhave26/TIGER/issues
- Check the main repository README for additional information
