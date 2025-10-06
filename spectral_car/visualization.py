"""
Visualization functions for spectral CAR models.
"""
import torch
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_spatial_field(
    values: torch.Tensor,
    grid_size: int,
    title: str = "Spatial Field",
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    """
    Plot spatial field on a grid.
    
    Args:
        values: Field values (n_obs,)
        grid_size: Size of square grid
        title: Plot title
        ax: Matplotlib axis (creates new if None)
        **kwargs: Additional arguments for imshow
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    field = values.reshape(grid_size, grid_size).cpu().numpy()
    im = ax.imshow(field, cmap='RdBu_r', **kwargs)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    
    return ax


def plot_spectral_density(
    eigenvalues: torch.Tensor,
    spectral_density: torch.Tensor,
    ax: Optional[plt.Axes] = None
):
    """
    Plot learned spectral density function.
    
    Args:
        eigenvalues: Graph Laplacian eigenvalues (n,)
        spectral_density: Precision at each eigenvalue (n,)
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(eigenvalues.cpu(), spectral_density.cpu(), 'b-', linewidth=2)
    ax.set_xlabel('Spatial Frequency (eigenvalue)')
    ax.set_ylabel('Precision p(λ)')
    ax.set_title('Learned Spectral Density')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_convergence(history: List[dict], metrics: List[str] = ['elbo']):
    """
    Plot convergence diagnostics.
    
    Args:
        history: List of diagnostic dictionaries from training
        metrics: Which metrics to plot
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = [h[metric] for h in history]
        iterations = [h['iteration'] for h in history]
        ax.plot(iterations, values)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Convergence')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig