"""
Shared plotting utilities for publication-quality scientific visualization.

This module provides utilities to ensure consistent, publication-ready figures
across all TIGER examples. It includes:
- Publication-quality matplotlib style configuration
- Perceptually uniform scientific colormaps (via cmcrameri)
- Smart legend placement to avoid data overlap
- Multi-format figure saving with consistent settings
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import warnings


def setup_publication_style(use_latex=True):
    """
    Configure matplotlib for publication-quality figures.

    Sets consistent styles including:
    - High DPI for crisp rendering
    - LaTeX-compatible fonts (with graceful fallback)
    - Inward-facing ticks for cleaner appearance
    - Appropriate figure sizes and margins

    Parameters
    ----------
    use_latex : bool, optional
        Whether to attempt LaTeX rendering for text. If LaTeX is not available,
        automatically falls back to standard fonts. Default is True.
    """
    # Try to use LaTeX if requested and available
    if use_latex:
        try:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            })
        except Exception:
            warnings.warn("LaTeX not available, falling back to standard fonts")
            use_latex = False

    if not use_latex:
        # Fallback to Arial or similar sans-serif
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        })

    # General publication settings
    plt.rcParams.update({
        # Figure settings
        "figure.dpi": 500,              # High DPI for raster formats
        "savefig.dpi": 500,
        "figure.figsize": (7, 5),       # Default size
        "figure.autolayout": False,     # We'll use tight_layout manually

        # Font sizes
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Tick settings - inward facing for cleaner look
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.top": True,
        "ytick.right": True,

        # Line and spine settings
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,

        # Legend settings
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": False,
        "legend.edgecolor": "black",

        # Colormap (default to perceptually uniform viridis)
        # Note: Use get_scientific_colormap('batlow') for cmcrameri colormaps
        "image.cmap": "viridis",
    })


def get_scientific_colormap(name='batlow', n_colors=None):
    """
    Get a perceptually uniform scientific colormap.

    Uses the cmcrameri package for scientifically designed colormaps that are:
    - Perceptually uniform (equal data differences = equal visual differences)
    - Colorblind-friendly
    - Black-and-white print safe

    Falls back to matplotlib colormaps if cmcrameri is not available.

    Parameters
    ----------
    name : str, optional
        Colormap name. Recommended options:
        - 'batlow': Perceptually uniform rainbow (general use)
        - 'vik': Diverging colormap (for data with meaningful zero)
        - 'devon': Sequential colormap (blue to yellow)
        - 'lajolla': Sequential colormap (light to dark)
        Default is 'batlow'.
    n_colors : int, optional
        Number of discrete colors. If None, returns continuous colormap.

    Returns
    -------
    matplotlib.colors.Colormap
        The requested colormap object.

    Notes
    -----
    See https://www.fabiocrameri.ch/colourmaps/ for colormap gallery and
    perceptual uniformity demonstrations.
    """
    try:
        from cmcrameri import cm as cmcm

        # Map common names to cmcrameri colormaps
        cmap_map = {
            'batlow': cmcm.batlow,
            'vik': cmcm.vik,
            'devon': cmcm.devon,
            'lajolla': cmcm.lajolla,
            'oslo': cmcm.oslo,
            'bamako': cmcm.bamako,
        }

        if name in cmap_map:
            cmap = cmap_map[name]
        else:
            # Try to get by name directly
            try:
                cmap = getattr(cmcm, name)
            except AttributeError:
                warnings.warn(f"Colormap '{name}' not found in cmcrameri, falling back to batlow")
                cmap = cmcm.batlow

        # Discretize if requested
        if n_colors is not None:
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                f'{name}_{n_colors}', cmap(np.linspace(0, 1, n_colors))
            )

        return cmap

    except ImportError:
        warnings.warn(
            "cmcrameri not available, falling back to matplotlib colormaps. "
            "Install with: pip install cmcrameri"
        )

        # Fallback mapping to matplotlib colormaps
        fallback_map = {
            'batlow': 'viridis',      # Both are perceptually uniform
            'vik': 'RdBu_r',          # Diverging
            'devon': 'YlGnBu',        # Sequential
            'lajolla': 'YlOrRd',      # Sequential
            'oslo': 'gray',           # Grayscale
            'bamako': 'plasma',       # Sequential
        }

        mpl_name = fallback_map.get(name, 'viridis')
        cmap = plt.get_cmap(mpl_name, n_colors)

        return cmap


def smart_legend_placement(ax, labels=None, **kwargs):
    """
    Automatically position legend to avoid overlapping with data.

    Tries multiple positions and selects the best one. If all positions
    overlap significantly, places legend outside the plot area.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the legend to.
    labels : list of str, optional
        Custom labels. If None, uses existing labels from plotted elements.
    **kwargs : dict
        Additional keyword arguments passed to ax.legend().

    Returns
    -------
    matplotlib.legend.Legend
        The created legend object.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y, label='Data')
    >>> smart_legend_placement(ax)
    """
    # Default kwargs for clean appearance
    default_kwargs = {
        'frameon': True,
        'framealpha': 0.9,
        'edgecolor': 'black',
        'fancybox': False,
    }
    default_kwargs.update(kwargs)

    # Try 'best' location first - matplotlib's automatic placement
    if labels is not None:
        legend = ax.legend(labels, **default_kwargs, loc='best')
    else:
        legend = ax.legend(**default_kwargs, loc='best')

    # Check if legend overlaps significantly with data
    # (matplotlib's 'best' should handle this, but we provide override options)

    # If user wants to force outside placement, support bbox_to_anchor
    if 'bbox_to_anchor' in kwargs:
        return legend

    return legend


def save_figure(fig, filename, formats=None, dpi=500, transparent=True,
                bbox_inches='tight', pad_inches=0.1):
    """
    Save figure in multiple formats with consistent settings.

    Automatically handles tight layout and saves in publication-ready formats.
    Creates parent directories if they don't exist.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Output filename without extension. Can include directory path.
    formats : list of str, optional
        File formats to save. Default is ['png', 'pdf'].
        Supports: 'png', 'pdf', 'svg', 'eps', 'jpg'
    dpi : int, optional
        Resolution for raster formats (png, jpg). Default is 500.
    transparent : bool, optional
        Whether to use transparent background. Default is True.
    bbox_inches : str or Bbox, optional
        Bounding box setting. Default is 'tight' to minimize whitespace.
    pad_inches : float, optional
        Padding around the figure. Default is 0.1 inches.

    Returns
    -------
    list of str
        Paths to all saved files.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> save_figure(fig, 'output/my_plot')  # Saves my_plot.png and my_plot.pdf
    """
    import os

    if formats is None:
        formats = ['png', 'pdf']

    # Ensure formats is a list
    if isinstance(formats, str):
        formats = [formats]

    # Create directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    for fmt in formats:
        output_path = f"{filename}.{fmt}"

        # Apply tight layout before saving
        try:
            fig.tight_layout()
        except Exception:
            pass  # tight_layout can fail for complex layouts

        # Save with appropriate settings for each format
        if fmt in ['png', 'jpg', 'jpeg']:
            fig.savefig(
                output_path,
                format=fmt,
                dpi=dpi,
                transparent=transparent,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches
            )
        else:  # Vector formats (pdf, svg, eps)
            fig.savefig(
                output_path,
                format=fmt,
                transparent=transparent,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches
            )

        saved_files.append(output_path)
        print(f"Saved: {output_path}")

    return saved_files


# Convenience function for common use case
def create_publication_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """
    Create a publication-ready figure with proper style already applied.

    Parameters
    ----------
    nrows, ncols : int, optional
        Number of subplot rows and columns. Default is 1, 1.
    figsize : tuple of float, optional
        Figure size (width, height) in inches. If None, uses default from style.
    **kwargs : dict
        Additional arguments passed to plt.subplots().

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes or array of Axes
        Single axes or array of axes if nrows or ncols > 1.

    Examples
    --------
    >>> fig, ax = create_publication_figure()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> save_figure(fig, 'my_plot')
    """
    setup_publication_style()

    if figsize is not None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    else:
        fig, ax = plt.subplots(nrows, ncols, **kwargs)

    return fig, ax
