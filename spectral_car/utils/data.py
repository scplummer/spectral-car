"""
Data Generation Utilities functions for spectral CAR models.

"""
from typing import Tuple, Optional, List

import torch
import numpy as np

# Compute spectral density
from spectral_car.utils.graph import create_grid_graph_laplacian


def generate_synthetic_spatial_data(
    n_obs: int,
    n_features: int,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    theta: Optional[torch.Tensor] = None,
    tau2: float = 0.5,
    sigma2: float = 0.25,
    seed: Optional[int] = None,
    spectral_form: str = 'rational'
) -> dict:
    """
    Generate synthetic spatial data with CAR structure.
    
    Creates data from the model:
        y = X*beta + phi + epsilon
        phi ~ CAR(theta, tau^2)
        epsilon ~ N(0, sigma^2)
    
    Args:
        n_obs: Number of observations
        n_features: Number of covariates (including intercept)
        eigenvalues: Graph Laplacian eigenvalues
        eigenvectors: Graph Laplacian eigenvectors
        beta: Fixed effects (if None, randomly generated)
        theta: Spectral parameters (if None, randomly generated)
               For 'rational': [log(a), log(b)] where p(λ) = 1/(a + b*λ)
               For 'exponential': [log(a), log(b)] where p(λ) = exp(-a - b*λ)
               For 'power_law': [log(a), log(b), log(d)] where p(λ) = 1/(a + b*λ)^d
        tau2: Spatial variance
        sigma2: Observation noise variance
        seed: Random seed for reproducibility
        spectral_form: Type of spectral density ('rational', 'exponential', 'power_law')
        
    Returns:
        Dictionary containing:
            - y: Observations (n_obs,)
            - X: Design matrix (n_obs, n_features)
            - phi: True spatial effects (n_obs,)
            - beta: True fixed effects (n_features,)
            - theta: True spectral coefficients
            - tau2: Spatial variance
            - sigma2: Observation noise variance
            - spectral_form: Type of spectral density used
            
    Example:
        >>> eigenvalues, eigenvectors = create_grid_graph_laplacian(64, 8)
        >>> data = generate_synthetic_spatial_data(
        ...     n_obs=64, n_features=3, 
        ...     eigenvalues=eigenvalues, eigenvectors=eigenvectors,
        ...     spectral_form='rational'
        ... )
        >>> print(data['y'].shape)  # torch.Size([64])
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate design matrix
    X = torch.randn(n_obs, n_features)
    X[:, 0] = 1.0  # Intercept
    
    # Generate or use provided beta
    if beta is None:
        beta = torch.randn(n_features)
    
    # Generate or use provided theta based on spectral form
    if theta is None:
        if spectral_form == 'rational':
            # Default: 1/(a + b*lambda) with moderate decay
            theta = torch.tensor([0.0, 1.0])  # a=1, b=e
        elif spectral_form == 'exponential':
            # Default: exp(-a - b*lambda)
            theta = torch.tensor([0.0, 1.0])
        elif spectral_form == 'power_law':
            # Default: 1/(a + b*lambda)^d
            theta = torch.tensor([0.0, 1.0, 0.5])  # d = sqrt(e) ≈ 1.65
        else:
            raise ValueError(f"Unknown spectral_form: {spectral_form}")
    
    # Normalize eigenvalues to [0, 1]
    lambda_min = eigenvalues.min()
    lambda_max = eigenvalues.max()
    eigenvalues_norm = (eigenvalues - lambda_min) / (lambda_max - lambda_min + 1e-8)
    
    # Compute spectral density based on chosen form
    if spectral_form == 'rational':
        # p(λ) = 1 / (a + b*λ)
        a = torch.exp(theta[0])
        b = torch.exp(theta[1])
        p_lambda = 1.0 / (a + b * eigenvalues_norm + 1e-8)
        
    elif spectral_form == 'exponential':
        # p(λ) = exp(-a - b*λ)
        a = torch.exp(theta[0])
        b = torch.exp(theta[1])
        p_lambda = torch.exp(-a - b * eigenvalues_norm)
        
    elif spectral_form == 'power_law':
        # p(λ) = 1 / (a + b*λ)^d
        a = torch.exp(theta[0])
        b = torch.exp(theta[1])
        d = torch.exp(theta[2])
        p_lambda = 1.0 / ((a + b * eigenvalues_norm + 1e-8) ** d)
        
    else:
        raise ValueError(f"Unknown spectral_form: {spectral_form}")
    
    # Generate spatial effect from CAR prior
    # Q^{-1} = U @ diag(1 / (tau^2 * p_lambda)) @ U^T
    Q_inv = eigenvectors @ torch.diag(1.0 / (tau2 * p_lambda)) @ eigenvectors.T
    
    # Sample phi ~ N(0, Q^{-1})
    phi = torch.distributions.MultivariateNormal(
        torch.zeros(n_obs), Q_inv
    ).sample()
    
    # Generate observations
    y = X @ beta + phi + torch.randn(n_obs) * np.sqrt(sigma2)
    
    return {
        'y': y,
        'X': X,
        'phi': phi,
        'beta': beta,
        'theta': theta,
        'tau2': tau2,
        'sigma2': sigma2,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'spectral_form': spectral_form,
    }

#===========================================
# Non - CAR Examples
#===========================================

def generate_smooth_spatial_field(
    grid_size: int,
    smoothness: float = 2.0,
    scale: float = 1.0,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate smooth spatial field using Gaussian process. 
    
    Creates a smooth random field on a 2D grid using a Matérn covariance.
    Useful for generating realistic spatial patterns.
    
    Args:
        grid_size: Size of square grid
        smoothness: Smoothness parameter (higher = smoother)
        scale: Overall scale of field
        seed: Random seed
        
    Returns:
        Spatial field (grid_size^2,)
        
    Example:
        >>> field = generate_smooth_spatial_field(8, smoothness=2.0)
        >>> field_2d = field.reshape(8, 8)
        >>> plt.imshow(field_2d)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    n_obs = grid_size ** 2
    
    # Create coordinates
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Compute distances
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    distances = torch.sqrt(torch.sum(diff**2, dim=2))
    
    # Matérn covariance (simplified)
    length_scale = 0.2 / smoothness
    K = torch.exp(-distances / length_scale)
    K = K + torch.eye(n_obs) * 1e-6  # Add jitter for numerical stability
    
    # Sample from GP
    field = torch.distributions.MultivariateNormal(
        torch.zeros(n_obs), K
    ).sample() * scale
    
    return field


def generate_multi_scale_field(
    grid_size: int,
    scales: List[float] = [0.5, 1.0, 2.0],
    weights: Optional[List[float]] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate spatial field with multiple scales.
    
    Combines multiple smooth fields at different scales to create
    complex spatial patterns. Useful for testing models' ability to
    capture multi-scale structure.
    
    Args:
        grid_size: Size of square grid
        scales: List of scale parameters (smaller = finer scale)
        weights: Weights for each scale (if None, equal weights)
        seed: Random seed
        
    Returns:
        Spatial field (grid_size^2,)
        
    Example:
        >>> # Create field with fine, medium, and coarse patterns
        >>> field = generate_multi_scale_field(8, scales=[0.5, 1.0, 2.0])
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if weights is None:
        weights = [1.0] * len(scales)
    
    weights = torch.tensor(weights)
    weights = weights / weights.sum()  # Normalize
    
    # Generate field at each scale
    field = torch.zeros(grid_size ** 2)
    for scale, weight in zip(scales, weights):
        field += weight * generate_smooth_spatial_field(
            grid_size, smoothness=scale, scale=1.0
        )
    
    return field


def generate_anisotropic_field(
    grid_size: int,
    angle: float = 0.0,
    aspect_ratio: float = 2.0,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate anisotropic spatial field.
    
    Creates a field with directional dependence (e.g., stronger correlation
    in one direction than another). Useful for testing models on realistic
    environmental data.
    
    Args:
        grid_size: Size of square grid
        angle: Angle of anisotropy in radians (0 = horizontal)
        aspect_ratio: Ratio of correlation lengths (> 1 = elongated)
        seed: Random seed
        
    Returns:
        Spatial field (grid_size^2,)
        
    Example:
        >>> # Create field elongated at 45 degrees
        >>> field = generate_anisotropic_field(8, angle=np.pi/4, aspect_ratio=3.0)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    n_obs = grid_size ** 2
    
    # Create coordinates
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    # Rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float)
    
    # Scale matrix (anisotropy)
    scale = torch.tensor([[1.0, 0.0], [0.0, 1.0 / aspect_ratio]], dtype=torch.float)
    
    # Transform matrix
    transform = rotation @ scale @ rotation.T
    
    # Compute anisotropic distances
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (n, n, 2)
    transformed_diff = torch.matmul(diff, transform)
    distances = torch.sqrt(torch.sum(transformed_diff**2, dim=2))
    
    # Covariance
    length_scale = 0.2
    K = torch.exp(-distances / length_scale)
    K = K + torch.eye(n_obs) * 1e-6
    
    # Sample
    field = torch.distributions.MultivariateNormal(
        torch.zeros(n_obs), K
    ).sample()
    
    return field


def add_outliers(
    y: torch.Tensor,
    proportion: float = 0.05,
    magnitude: float = 5.0,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add outliers to data.
    
    Randomly corrupts a proportion of observations with large errors.
    Useful for testing robustness of models.
    
    Args:
        y: Original observations (n,)
        proportion: Proportion of outliers (0 to 1)
        magnitude: Size of outliers in standard deviations
        seed: Random seed
        
    Returns:
        y_corrupted: Observations with outliers (n,)
        outlier_mask: Boolean mask indicating outliers (n,)
        
    Example:
        >>> y_clean = torch.randn(100)
        >>> y_outliers, mask = add_outliers(y_clean, proportion=0.1, magnitude=5.0)
        >>> print(f"Added {mask.sum()} outliers")
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    n = len(y)
    n_outliers = int(n * proportion)
    
    # Random outlier locations
    outlier_indices = torch.randperm(n)[:n_outliers]
    outlier_mask = torch.zeros(n, dtype=torch.bool)
    outlier_mask[outlier_indices] = True
    
    # Add outliers
    y_corrupted = y.clone()
    outlier_std = torch.std(y) * magnitude
    y_corrupted[outlier_mask] += torch.randn(n_outliers) * outlier_std
    
    return y_corrupted, outlier_mask


def generate_train_test_split(
    n_obs: int,
    grid_size: int,
    test_proportion: float = 0.2,
    split_type: str = 'random',
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate train/test split for spatial data.
    
    Args:
        n_obs: Total number of observations
        grid_size: Size of square grid (for spatial splits)
        test_proportion: Proportion of data for testing
        split_type: Type of split ('random', 'spatial_block', 'checkerboard')
        seed: Random seed
        
    Returns:
        train_indices: Indices for training (n_train,)
        test_indices: Indices for testing (n_test,)
        
    Example:
        >>> train_idx, test_idx = generate_train_test_split(
        ...     64, 8, test_proportion=0.2, split_type='spatial_block'
        ... )
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if split_type == 'random':
        # Random split
        indices = torch.randperm(n_obs)
        n_test = int(n_obs * test_proportion)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
    elif split_type == 'spatial_block':
        # Hold out a spatial block for testing
        n_test = int(grid_size * test_proportion)
        test_mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)
        test_mask[:n_test, :] = True  # Top block
        
        test_indices = torch.where(test_mask.flatten())[0]
        train_indices = torch.where(~test_mask.flatten())[0]
        
    elif split_type == 'checkerboard':
        # Checkerboard pattern
        test_mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 0:
                    test_mask[i, j] = True
        
        # Subsample to get desired proportion
        test_candidates = torch.where(test_mask.flatten())[0]
        n_test = int(n_obs * test_proportion)
        test_indices = test_candidates[torch.randperm(len(test_candidates))[:n_test]]
        train_indices = torch.tensor([i for i in range(n_obs) if i not in test_indices])
        
    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    
    return train_indices, test_indices

#====================================
# Generate Benchmark Datasets
#====================================



def generate_benchmark_dataset(
    name: str,
    n_obs: int,
    grid_size: int,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    n_features: int = 3,
    seed: Optional[int] = None
) -> dict:
    """
    Generate standard benchmark datasets for testing.
    
    Provides pre-configured synthetic datasets with different spatial
    characteristics for consistent model testing and comparison.
    
    Args:
        name: Dataset name ('smooth', 'rough', 'multi_scale', 'anisotropic')
        n_obs: Number of observations
        grid_size: Size of spatial grid (for reshaping)
        eigenvalues: Graph Laplacian eigenvalues
        eigenvectors: Graph Laplacian eigenvectors
        n_features: Number of covariate features (default: 3)
        seed: Random seed for reproducibility
        
    Returns:
        data: Dictionary with keys:
            - y: Observations (n_obs,)
            - X: Covariates (n_obs, n_features)
            - phi: True spatial field (n_obs,)
            - beta: True coefficients (n_features,)
            - theta: True spectral parameters (poly_order,) or None
            - tau2: True spatial variance
            - sigma2: True noise variance
            - eigenvalues: Eigenvalues (same as input)
            - eigenvectors: Eigenvectors (same as input)
            - name: Dataset name
            
    Available datasets:
        - 'smooth': Smooth spatial field with strong spatial correlation
        - 'rough': Rough spatial field with weak spatial correlation
        - 'multi_scale': Multiple spatial scales
        - 'anisotropic': Directional spatial patterns
        
    Example:
        >>> eigenvalues, eigenvectors = create_grid_graph_laplacian(100, 10)
        >>> data = generate_benchmark_dataset(
        ...     'smooth', 
        ...     n_obs=100, 
        ...     grid_size=10,
        ...     eigenvalues=eigenvalues,
        ...     eigenvectors=eigenvectors,
        ...     seed=42
        ... )
        >>> y, X, phi = data['y'], data['X'], data['phi']
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    n_obs = grid_size ** 2
    eigenvalues, eigenvectors = create_grid_graph_laplacian(n_obs, grid_size)
    
    if name == 'smooth':
        # Strong spatial correlation (slow decay)
        theta = torch.tensor([0.0, 0.5])  # a=1, b=1.65
        spectral_form = 'rational'
        tau2, sigma2 = 1.0, 0.1
        
    elif name == 'rough':
        # Weak spatial correlation (fast decay)
        theta = torch.tensor([0.0, 2.0])  # a=1, b=7.4
        spectral_form = 'rational'
        tau2, sigma2 = 0.3, 0.5
        
    elif name == 'multi_scale':
        # Could use power_law for more flexibility
        theta = torch.tensor([0.0, 1.0, 0.7])  # a=1, b=2.7, d=2.0
        spectral_form = 'power_law'
        tau2, sigma2 = 0.5, 0.25
        
    elif name == 'anisotropic':
        # Generate anisotropic field directly
        phi = generate_anisotropic_field(grid_size, angle=np.pi/4, aspect_ratio=3.0)
        
        # Generate other components
        X = torch.randn(n_obs, 3)
        X[:, 0] = 1.0
        beta = torch.tensor([2.0, -1.0, 0.5])
        sigma2 = 0.25
        y = X @ beta + phi + torch.randn(n_obs) * np.sqrt(sigma2)
        
        return {
            'y': y,
            'X': X,
            'phi': phi,
            'beta': beta,
            'theta': None,  # Not applicable
            'tau2': None,
            'sigma2': sigma2,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'name': name
        }
        
    else:
        raise ValueError(f"Unknown benchmark dataset: {name}")
    
    # Generate data using specified parameters
    data = generate_synthetic_spatial_data(
        n_obs=n_obs,
        n_features=3,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        beta=torch.tensor([2.0, -1.0, 0.5]),
        theta=theta,
        tau2=tau2,
        sigma2=sigma2,
        spectral_form=spectral_form,
        seed=seed
    )
    data['name'] = name
    data['spectral_form'] = spectral_form

    return data