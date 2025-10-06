"""
Spectral CAR: Spectral Conditional Autoregressive Models with Variational Inference

A Python package for Bayesian spatial modeling using spectral representations
of conditional autoregressive (CAR) models with variational inference.

Main Components:
    - models: Spectral CAR model classes
    - inference: Variational inference engines
    - utils: Utility functions for data generation, diagnostics, etc.
    - visualization: Plotting and diagnostic visualizations
"""

__version__ = "0.1.0"
__author__ = "Sean Plummer"
__license__ = "MIT"

# Core models
from .models import (
    SpectralCARBase,
    SpectralCARMeanField,
    SpectralCARLowRank,
    #TODO: SpectralCARCollapsed,
)

# Inference engines
from .inference import (
    VariationalInference,
    CalibratedVI,
)

# Most commonly used utilities (convenient imports)
from .utils import (
    # Graph utilities
    create_grid_graph_laplacian,
    create_adjacency_from_coords,
    # Data generation
    generate_synthetic_spatial_data,
    generate_benchmark_dataset,
    # Diagnostics
    compute_waic,
    compute_loo,
    compare_models,
    compute_prediction_intervals,
    compute_coverage,
    # Calibration
    find_calibration_factor,
)

# Visualization (import the module, not individual functions)
from . import visualization

__all__ = [
    # Version info
    '__version__',
    # Models
    'SpectralCARBase',
    'SpectralCARMeanField',
    'SpectralCARLowRank',
    'SpectralCARCollapsed',
    # Inference
    'VariationalInference',
    'CalibratedVI',
    # Utils - Graph
    'create_grid_graph_laplacian',
    'create_adjacency_from_coords',
    # Utils - Data
    'generate_synthetic_spatial_data',
    'generate_benchmark_dataset',
    # Utils - Diagnostics
    'compute_waic',
    'compute_loo',
    'compare_models',
    'compute_prediction_intervals',
    'compute_coverage',
    # Submodules
    'visualization',
]


# Package-level configuration
def get_config():
    """Get current package configuration."""
    return {
        'version': __version__,
        'author': __author__,
        'license': __license__,
    }


def print_info():
    """Print package information."""
    print(f"Spectral CAR v{__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print("\nAvailable models:")
    print("  - SpectralCARMeanField: Mean-field variational inference")
    print("  - SpectralCARLowRank: Low-rank approximation")
    #TODO: print("  - SpectralCARCollapsed: Marginalized spatial effects")
    print("\nFor documentation, see: https://github.com/scplummer/spectral-car")