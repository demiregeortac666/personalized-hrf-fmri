"""
Visualization utilities for HRF parameters and simulation results.

This module provides functions for visualizing HRF parameters, hemodynamic 
responses, and simulation results using various plotting techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors


def plot_hrf(t, hrf, title="Hemodynamic Response Function", ax=None, color='blue', 
             linestyle='-', label=None, alpha=1.0):
    """
    Plot a hemodynamic response function.
    
    Parameters
    ----------
    t : array-like
        Time points
    hrf : array-like
        HRF values
    title : str, optional
        Plot title, by default "Hemodynamic Response Function"
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (creates new figure)
    color : str, optional
        Line color, by default 'blue'
    linestyle : str, optional
        Line style, by default '-'
    label : str, optional
        Line label for legend, by default None
    alpha : float, optional
        Line transparency, by default 1.0
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(t, hrf, color=color, linestyle=linestyle, label=label, alpha=alpha)
    
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if label is not None:
        ax.legend()
    
    return ax


def plot_region_hrfs(region_hrfs, region_names=None, figsize=(15, 10), n_cols=3):
    """
    Plot HRFs for multiple brain regions.
    
    Parameters
    ----------
    region_hrfs : dict
        Dictionary of region HRFs as returned by HRFEstimator.estimate_region_specific_hrfs()
    region_names : dict or list, optional
        Dictionary or list of region names, by default None (uses region IDs)
    figsize : tuple, optional
        Figure size, by default (15, 10)
    n_cols : int, optional
        Number of columns in the plot grid, by default 3
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plots
    """
    n_regions = len(region_hrfs)
    n_rows = int(np.ceil(n_regions / n_cols))
    
    fig = plt.figure(figsize=figsize)
    
    for i, (region_id, (t, hrf, hrf_fitted)) in enumerate(region_hrfs.items()):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Plot estimated HRF
        plot_hrf(t, hrf, title="", ax=ax, color='blue', 
                 linestyle='-', label="Estimated", alpha=0.7)
        
        # Plot fitted HRF from Balloon-Windkessel model
        plot_hrf(t, hrf_fitted, title="", ax=ax, color='red', 
                 linestyle='--', label="Model Fit", alpha=0.7)
        
        # Set title with region name if available
        if region_names is not None:
            if isinstance(region_names, dict):
                region_name = region_names.get(region_id, f"Region {region_id}")
            else:
                region_name = region_names[region_id] if i < len(region_names) else f"Region {region_id}"
        else:
            region_name = f"Region {region_id}"
            
        ax.set_title(region_name)
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    return fig


def plot_balloon_windkessel_states(t, states, region_id=0, title=None, figsize=(12, 8)):
    """
    Plot the state variables of the Balloon-Windkessel model.
    
    Parameters
    ----------
    t : array-like
        Time points
    states : array-like
        State variables [s, f, v, q] with shape (4, time_points)
    region_id : int or str, optional
        Region identifier for the title, by default 0
    title : str, optional
        Plot title, by default None (creates a default title)
    figsize : tuple, optional
        Figure size, by default (12, 8)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plots
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()
    
    # State labels and colors
    labels = ['Signal (s)', 'Flow (f)', 'Volume (v)', 'Deoxyhemoglobin (q)']
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax = axs[i]
        ax.plot(t, states[i], color=color)
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.grid(True, linestyle='--', alpha=0.7)
    
    if title is None:
        title = f"Balloon-Windkessel Model States for Region {region_id}"
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def visualize_brain_parameters(params, brain_regions, param_name, cmap='viridis', 
                               figsize=(12, 8), title=None, vmin=None, vmax=None):
    """
    Visualize brain parameters across regions.
    
    Parameters
    ----------
    params : dict
        Dictionary of parameter values for each region
    brain_regions : array-like
        Array of region coordinates (x, y, z) or 3D brain volume
    param_name : str
        Name of the parameter to visualize
    cmap : str, optional
        Colormap name, by default 'viridis'
    figsize : tuple, optional
        Figure size, by default (12, 8)
    title : str, optional
        Plot title, by default None (creates a default title)
    vmin : float, optional
        Minimum value for colormap scaling, by default None
    vmax : float, optional
        Maximum value for colormap scaling, by default None
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    # Extract the parameter values for each region
    param_values = np.array([params[region_id].get(param_name, np.nan) 
                            for region_id in params.keys()])
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Different handling based on the type of brain_regions
    if isinstance(brain_regions, np.ndarray) and brain_regions.ndim == 2:
        # If brain_regions is a list of coordinates
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create 3D scatter plot
        ax = fig.add_subplot(111, projection='3d')
        
        # Create colormap
        norm = mcolors.Normalize(vmin=vmin or np.nanmin(param_values), 
                                vmax=vmax or np.nanmax(param_values))
        
        # Plot brain regions as spheres colored by parameter value
        scatter = ax.scatter(brain_regions[:, 0], brain_regions[:, 1], brain_regions[:, 2],
                           c=param_values, cmap=cmap, norm=norm, alpha=0.7, s=100)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    else:
        # If brain_regions is something else, create a flat visualization
        # This would be extended with actual neuroimaging visualization for real applications
        ax = fig.add_subplot(111)
        
        # Create a simple visualization (e.g., a bar plot)
        bar = ax.bar(range(len(param_values)), param_values, color=plt.cm.get_cmap(cmap)(
            (param_values - np.nanmin(param_values)) / (np.nanmax(param_values) - np.nanmin(param_values))
        ))
        
        ax.set_xlabel('Region ID')
        ax.set_ylabel(param_name)
        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels(list(params.keys()), rotation=90)
    
    # Add colorbar
    plt.colorbar(scatter if 'scatter' in locals() else plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap(cmap)), 
               ax=ax, label=param_name)
    
    if title is None:
        title = f"Distribution of {param_name} Across Brain Regions"
        
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_bold_signals(time, bold_signals, region_ids=None, region_names=None, 
                     figsize=(12, 8), n_cols=2, title="BOLD Signals"):
    """
    Plot BOLD signals for multiple brain regions.
    
    Parameters
    ----------
    time : array-like
        Time points
    bold_signals : array-like
        BOLD signals with shape (time_points, regions)
    region_ids : list, optional
        List of region IDs, by default None (uses indices)
    region_names : dict or list, optional
        Dictionary or list of region names, by default None (uses region IDs)
    figsize : tuple, optional
        Figure size, by default (12, 8)
    n_cols : int, optional
        Number of columns in the plot grid, by default 2
    title : str, optional
        Overall plot title, by default "BOLD Signals"
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plots
    """
    if bold_signals.ndim != 2:
        raise ValueError("BOLD signals should be 2D with shape (time_points, regions)")
    
    n_regions = bold_signals.shape[1]
    
    if region_ids is None:
        region_ids = list(range(n_regions))
    
    n_rows = int(np.ceil(n_regions / n_cols))
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    for i, region_id in enumerate(region_ids):
        if i >= n_regions:
            break
            
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        
        ax.plot(time, bold_signals[:, i], 'b-')
        
        # Set title with region name if available
        if region_names is not None:
            if isinstance(region_names, dict):
                region_name = region_names.get(region_id, f"Region {region_id}")
            else:
                region_name = region_names[region_id] if i < len(region_names) else f"Region {region_id}"
        else:
            region_name = f"Region {region_id}"
            
        ax.set_title(region_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("BOLD Signal")
        ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig


def plot_comparison(time1, signal1, time2, signal2, label1="Signal 1", label2="Signal 2",
                   title="Signal Comparison", figsize=(10, 6)):
    """
    Plot a comparison between two signals.
    
    Parameters
    ----------
    time1 : array-like
        Time points for signal 1
    signal1 : array-like
        Values for signal 1
    time2 : array-like
        Time points for signal 2
    signal2 : array-like
        Values for signal 2
    label1 : str, optional
        Label for signal 1, by default "Signal 1"
    label2 : str, optional
        Label for signal 2, by default "Signal 2"
    title : str, optional
        Plot title, by default "Signal Comparison"
    figsize : tuple, optional
        Figure size, by default (10, 6)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(time1, signal1, 'b-', label=label1)
    ax.plot(time2, signal2, 'r--', label=label2)
    
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    return fig 