"""
Utility functions for Spectral CAR models.

This subpackage contains helper functions organized into modules:
    - graph: Graph construction and Laplacian computation
    - spectral: Spectral transformations and density computation
    - calibration: Uncertainty calibration utilities
    - data: Synthetic data generation
    - diagnostics: Model diagnostics and comparison
"""

# Graph utilities
from .graph import (
    create_grid_graph_laplacian,
    create_adjacency_from_coords,
    create_laplacian_from_adjacency,
    add_graph_boundary,
)

# Spectral utilities
from .spectral import (
    chebyshev_polynomials,
)

# Calibration utilities
from .calibration import (
    find_calibration_factor,
)

# Data generation
from .data import (
    generate_synthetic_spatial_data,
    generate_smooth_spatial_field,
    generate_multi_scale_field,
    generate_anisotropic_field,
    generate_benchmark_dataset,
    add_outliers,
    generate_train_test_split,
)

# Diagnostics
from .diagnostics import (
    compute_effective_sample_size,
    compute_waic,
    compute_loo,
    compare_models,
    check_convergence_diagnostics,
    compute_prediction_intervals,
    compute_coverage,
    compute_spatial_autocorrelation,
)

__all__ = [
    # Graph
    'create_grid_graph_laplacian',
    'create_adjacency_from_coords',
    'create_laplacian_from_adjacency',
    'add_graph_boundary',
    # Spectral
    'chebyshev_polynomials',
    'compute_spectral_density',
    'normalize_eigenvalues',
    # Calibration
    'find_calibration_factor',
    'compute_calibration_coverage',
    'optimize_calibration_factor',
    # Data
    'generate_synthetic_spatial_data',
    'generate_smooth_spatial_field',
    'generate_multi_scale_field',
    'generate_anisotropic_field',
    'generate_benchmark_dataset',
    'add_outliers',
    'generate_train_test_split',
    # Diagnostics
    'compute_effective_sample_size',
    'compute_waic',
    'compute_loo',
    'compare_models',
    'check_convergence_diagnostics',
    'compute_prediction_intervals',
    'compute_coverage',
    'compute_spatial_autocorrelation',
]