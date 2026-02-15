# TIGER

TIGER is a Python module for directly accessing Exodus and Nemesis file data as numpy arrays. Examples demonstrate how to generate high quality figures using matplotlib and the TIGER interface.

## Querying variables
When reading multiple Exodus files, ``ExodusReader`` combines available variable
names from each file. The resulting lists are exposed on the multi-file reader
via ``nodal_var_names`` and ``elem_var_names``. These attributes show all nodal
and elemental variables that may be requested with APIs such as
``get_data_at_time``. These APIs also accept an optional ``block_id`` to
restrict queries to a specific element block. For example::

    x, y, z, c = reader.get_data_at_time('c_Cr', time, block_id=1)

This returns coordinates and values only for block ``1``.

## Quick Start

Clone the repository and run the setup script:

```bash
git clone https://github.com/chaitanyaBhave26/TIGER.git
cd TIGER
bash setup.sh
```

Activate the environment (required each time you use TIGER):

```bash
source tiger_env/bin/activate
```

That's it! TIGER is now ready to use.

## Installation Details

The `setup.sh` script:
1. Creates a Python virtual environment (`tiger_env/`)
2. Installs all required dependencies
3. Installs TIGER in development mode

**Manual installation** (if you prefer):
```bash
python3 -m venv tiger_env
source tiger_env/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Optional dependencies** (for video generation):
```bash
pip install -r requirements-optional.txt
```

## Context Manager Usage (Recommended)

For proper resource management, use ExodusReader as a context manager:

```python
with ExodusReader(filenames) as reader:
    times = reader.global_times
    x, y, z, c = reader.get_data_at_time('temperature', times[-1])
    # Process data...
# Reader is automatically closed
```

This ensures the underlying netCDF dataset is properly closed even if errors occur.

## Working with Multiple Element Blocks

For meshes with multiple element blocks (e.g., multi-material simulations):

```python
with ExodusReader('mesh.e') as reader:
    # See available blocks
    print(f"Blocks: {list(reader.block_connect.keys())}")

    # Extract data from specific block
    x, y, z, c = reader.get_data_at_time('stress', time, block_id=1)

    # Compare across blocks
    for block_id in reader.block_connect.keys():
        x, y, z, c = reader.get_data_at_time('temperature', time, block_id=block_id)
        # Analyze block-specific data...
```

The `block_id` parameter restricts queries to a specific element block. If omitted, data from all blocks is returned.

## Running Examples

Example scripts are located in the `examples/` directory. Before running:

1. Activate the virtual environment: `source tiger_env/bin/activate`
2. Create subdirectories for output: `mkdir -p examples/{1D,2D,3D}`
3. Place your Exodus data files in the appropriate directories
4. Run examples: `python examples/1d_plot.py`

See `examples/README.md` for detailed information about each example.

## Deactivating the Environment

When done working with TIGER:

```bash
deactivate
```

## Alternative Installation Methods

### Option A - Conda installation:
If you prefer Conda over venv:

1. Install Conda from the [Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create the TIGER environment:
    ```bash
    conda create --name tiger_env "h5py=*=mpi*" "netcdf4=1.6.*=mpi*" matplotlib scipy numpy pytest cmcrameri mpi4py -c conda-forge
    ```
3. Activate and install TIGER:
    ```bash
    conda activate tiger_env
    cd TIGER
    pip install -e .
    ```

### Option B - HPC with module system:
For HPC systems with environment modules:

1. Load required modules:
    ```bash
    module load python
    module load mpi
    ```
2. Follow the manual installation steps above
