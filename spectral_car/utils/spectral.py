"""
Spectral Transformation Utilities for spectral CAR models.
"""

from typing import Tuple, List

import torch

def chebyshev_polynomials(
    x: torch.Tensor, 
    order: int
) -> torch.Tensor:
    """
    Compute Chebyshev polynomials T_0(x), ..., T_K(x).
    
    Uses the three-term recurrence relation:
        T_0(x) = 1
        T_1(x) = x
        T_{k+1}(x) = 2x*T_k(x) - T_{k-1}(x)
    
    Args:
        x: Input values, shape (n,). Should be in [-1, 1] for stability.
        order: Maximum polynomial order K
        
    Returns:
        Tensor of shape (n, K+1) where [:, k] contains T_k(x)
        
    Example:
        >>> x = torch.linspace(-1, 1, 100)
        >>> T = chebyshev_polynomials(x, order=5)
        >>> T.shape
        torch.Size([100, 6])
    """
    n = x.shape[0]
    T = torch.zeros(n, order + 1, device=x.device, dtype=x.dtype)
    
    # Base cases
    T[:, 0] = 1.0
    if order >= 1:
        T[:, 1] = x
    
    # Recurrence relation
    for k in range(2, order + 1):
        T[:, k] = 2 * x * T[:, k-1] - T[:, k-2]
    
    return T


def normalize_eigenvalues(
    eigenvalues: torch.Tensor,
    method: str = 'chebyshev'
) -> Tuple[torch.Tensor, dict]:
    """
    Normalize eigenvalues for numerical stability.
    
    Args:
        eigenvalues: Raw eigenvalues (n,)
        method: Normalization method
            - 'chebyshev': Map to [-1, 1] for Chebyshev polynomials
            - 'unit': Map to [0, 1]
            - 'standardize': Z-score normalization
        
    Returns:
        eigenvalues_normalized: Normalized eigenvalues (n,)
        normalization_params: Dictionary with normalization parameters
        
    Example:
        >>> eigenvalues = torch.tensor([0.0, 0.5, 1.2, 2.3, 4.0])
        >>> eig_norm, params = normalize_eigenvalues(eigenvalues)
    """
    lambda_min = eigenvalues.min()
    lambda_max = eigenvalues.max()
    
    if method == 'chebyshev':
        # Map to [-1, 1]
        eigenvalues_normalized = 2 * (eigenvalues - lambda_min) / (lambda_max - lambda_min + 1e-8) - 1
        params = {'min': lambda_min, 'max': lambda_max, 'method': 'chebyshev'}
        
    elif method == 'unit':
        # Map to [0, 1]
        eigenvalues_normalized = (eigenvalues - lambda_min) / (lambda_max - lambda_min + 1e-8)
        params = {'min': lambda_min, 'max': lambda_max, 'method': 'unit'}
        
    elif method == 'standardize':
        # Z-score
        mean = eigenvalues.mean()
        std = eigenvalues.std()
        eigenvalues_normalized = (eigenvalues - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std, 'method': 'standardize'}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return eigenvalues_normalized, params


def spectral_density(
    eigenvalues: torch.Tensor,
    theta: torch.Tensor,
    spectral_form: str = 'rational'
) -> torch.Tensor:
    """
    Evaluate spectral density p(lambda; theta) using specified functional form.
    
    Args:
        eigenvalues: Eigenvalues to evaluate at (n,). Should be normalized to 
                    [0,1] for parametric forms or [-1,1] for Chebyshev.
        theta: Spectral parameters
               - 'chebyshev': (K+1,) polynomial coefficients
               - 'rational': (2,) = [log(a), log(b)] for p = 1/(a + b*λ)
               - 'exponential': (2,) = [log(a), log(b)] for p = exp(-a - b*λ)
               - 'power_law': (3,) = [log(a), log(b), log(d)] for p = 1/(a + b*λ)^d
        spectral_form: Type of spectral density function
        
    Returns:
        Spectral density values (n,) or (batch, n) if theta is batched
        
    Example:
        >>> eigenvalues = torch.linspace(0, 1, 100)
        >>> theta = torch.tensor([0.0, 1.0])  # a=1, b=e
        >>> p_lambda = spectral_density(eigenvalues, theta, 'rational')
    """
    if spectral_form == 'chebyshev':
        # Use polynomial approach
        order = theta.shape[-1] - 1
        T = chebyshev_polynomials(eigenvalues, order)
        
        if theta.dim() == 1:
            log_p = torch.matmul(T, theta)
        else:
            log_p = torch.matmul(T, theta.T).T
        
        spectral_density = torch.exp(log_p)
        
    elif spectral_form == 'rational':
        # p(λ) = 1 / (a + b*λ)
        a = torch.exp(theta[..., 0])
        b = torch.exp(theta[..., 1])
        
        if theta.dim() == 1:
            spectral_density = 1.0 / (a + b * eigenvalues + 1e-8)
        else:
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)
            spectral_density = 1.0 / (a + b * eigenvalues + 1e-8)
        
    elif spectral_form == 'exponential':
        # p(λ) = exp(-a - b*λ)
        a = torch.exp(theta[..., 0])
        b = torch.exp(theta[..., 1])
        
        if theta.dim() == 1:
            spectral_density = torch.exp(-a - b * eigenvalues)
        else:
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)
            spectral_density = torch.exp(-a - b * eigenvalues)
        
    elif spectral_form == 'power_law':
        # p(λ) = 1 / (a + b*λ)^d
        a = torch.exp(theta[..., 0])
        b = torch.exp(theta[..., 1])
        d = torch.exp(theta[..., 2])
        
        if theta.dim() == 1:
            spectral_density = 1.0 / ((a + b * eigenvalues + 1e-8) ** d)
        else:
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)
            d = d.unsqueeze(-1)
            spectral_density = 1.0 / ((a + b * eigenvalues + 1e-8) ** d)
    
    else:
        raise ValueError(f"Unknown spectral_form: {spectral_form}")
    
    # Clamp for numerical stability
    spectral_density = torch.clamp(spectral_density, min=1e-6, max=1e6)
    return spectral_density


def legendre_polynomials(
    x: torch.Tensor,
    order: int
) -> torch.Tensor:
    """
    Compute Legendre polynomials P_0(x), ..., P_K(x).
    
    Uses the recurrence relation:
        P_0(x) = 1
        P_1(x) = x
        (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)
    
    Args:
        x: Input values (n,). Should be in [-1, 1].
        order: Maximum polynomial order K
        
    Returns:
        Tensor of shape (n, K+1)
    """
    n = x.shape[0]
    P = torch.zeros(n, order + 1, device=x.device, dtype=x.dtype)
    
    # Base cases
    P[:, 0] = 1.0
    if order >= 1:
        P[:, 1] = x
    
    # Recurrence relation
    for k in range(1, order):
        P[:, k+1] = ((2*k + 1) * x * P[:, k] - k * P[:, k-1]) / (k + 1)
    
    return P


def spectral_transform(
    phi: torch.Tensor,
    eigenvectors: torch.Tensor,
    inverse: bool = False
) -> torch.Tensor:
    """
    Transform between spatial and spectral domains.
    
    Forward transform: alpha = U^T @ phi (spatial -> spectral)
    Inverse transform: phi = U @ alpha (spectral -> spatial)
    
    Args:
        phi: Spatial field (n,) or alpha spectral coefficients (n,)
        eigenvectors: Eigenvector matrix U (n, n)
        inverse: If False, compute U^T @ phi. If True, compute U @ alpha.
        
    Returns:
        Transformed values (n,)
        
    Example:
        >>> phi = torch.randn(64)
        >>> U = torch.randn(64, 64)
        >>> alpha = spectral_transform(phi, U, inverse=False)  # Spatial -> spectral
        >>> phi_reconstructed = spectral_transform(alpha, U, inverse=True)  # Spectral -> spatial
    """
    if inverse:
        # Spectral -> Spatial: phi = U @ alpha
        return eigenvectors @ phi
    else:
        # Spatial -> Spectral: alpha = U^T @ phi
        return eigenvectors.T @ phi


def compute_effective_range(
    eigenvalues: torch.Tensor,
    spectral_density: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute effective spatial range from spectral density.
    
    Finds the eigenvalue where spectral density drops below threshold times
    its maximum. This corresponds to the spatial scale where correlation
    becomes weak.
    
    Args:
        eigenvalues: Eigenvalues (n,)
        spectral_density: Precision values p(lambda) (n,)
        threshold: Threshold relative to max precision (default: 0.5)
        
    Returns:
        Effective range (in units of eigenvalues)
        
    Example:
        >>> eigenvalues = torch.linspace(0, 4, 100)
        >>> p_lambda = torch.exp(-eigenvalues)  # Exponential decay
        >>> effective_range = compute_effective_range(eigenvalues, p_lambda)
    """
    # Normalize spectral density
    p_normalized = spectral_density / spectral_density.max()
    
    # Find where it drops below threshold
    above_threshold = p_normalized >= threshold
    
    if above_threshold.sum() == 0:
        return eigenvalues[0].item()
    
    # Last eigenvalue above threshold
    indices = torch.where(above_threshold)[0]
    effective_lambda = eigenvalues[indices[-1]].item()
    
    return effective_lambda


def decompose_field_by_scale(
    phi: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    scales: List[Tuple[float, float]]
) -> List[torch.Tensor]:
    """
    Decompose spatial field into different scales.
    
    Separates field into low, mid, and high frequency components based
    on eigenvalue ranges.
    
    Args:
        phi: Spatial field (n,)
        eigenvalues: Eigenvalues (n,)
        eigenvectors: Eigenvectors (n, n)
        scales: List of (lambda_min, lambda_max) tuples defining scales
                Example: [(0, 0.5), (0.5, 2.0), (2.0, inf)]
        
    Returns:
        List of spatial fields, one per scale
        
    Example:
        >>> phi = torch.randn(64)
        >>> eigenvalues, eigenvectors = create_grid_graph_laplacian(64, 8)
        >>> scales = [(0, 1.0), (1.0, 3.0), (3.0, float('inf'))]
        >>> phi_low, phi_mid, phi_high = decompose_field_by_scale(
        ...     phi, eigenvalues, eigenvectors, scales
        ... )
    """
    # Transform to spectral domain
    alpha = spectral_transform(phi, eigenvectors, inverse=False)
    
    fields = []
    for lambda_min, lambda_max in scales:
        # Select components in this scale
        mask = (eigenvalues >= lambda_min) & (eigenvalues < lambda_max)
        alpha_scale = alpha.clone()
        alpha_scale[~mask] = 0
        
        # Transform back to spatial domain
        phi_scale = spectral_transform(alpha_scale, eigenvectors, inverse=True)
        fields.append(phi_scale)
    
    return fields


def spectral_smoothness(
    alpha: torch.Tensor,
    eigenvalues: torch.Tensor
) -> float:
    """
    Compute smoothness measure of a spatial field in spectral domain.
    
    Measures how much energy is in high vs low frequencies.
    Higher values indicate smoother fields.
    
    Args:
        alpha: Spectral coefficients (n,)
        eigenvalues: Eigenvalues (n,)
        
    Returns:
        Smoothness score (higher = smoother)
        
    Example:
        >>> alpha = torch.randn(64)
        >>> eigenvalues = torch.linspace(0, 4, 64)
        >>> smoothness = spectral_smoothness(alpha, eigenvalues)
    """
    # Compute spectral energy at each frequency
    energy = alpha ** 2
    
    # Weight by inverse eigenvalue (low frequencies get high weight)
    weights = 1.0 / (eigenvalues + 1.0)  # +1 to avoid division by zero
    
    # Smoothness = weighted average
    smoothness = torch.sum(weights * energy) / torch.sum(energy)
    
    return smoothness.item()


def estimate_spatial_range(
    eigenvalues: torch.Tensor,
    theta: torch.Tensor,
    spectral_form: str = 'rational',
    grid_spacing: float = 1.0
) -> float:
    """
    Estimate effective spatial correlation range from spectral density.
    
    Converts spectral density to approximate spatial correlation range.
    Useful for interpreting learned spectral filters.
    
    Args:
        eigenvalues: Eigenvalues (n,), should be normalized appropriately
        theta: Spectral parameters (depends on spectral_form)
        spectral_form: Type of spectral density ('rational', 'chebyshev', etc.)
        grid_spacing: Physical distance between grid points (default: 1.0)
        
    Returns:
        Estimated correlation range in physical units
        
    Example:
        >>> eigenvalues = torch.linspace(0, 1, 100)
        >>> theta = torch.tensor([0.0, 1.0])
        >>> range_est = estimate_spatial_range(eigenvalues, theta, 'rational')
    """
    # Compute spectral density
    p_lambda = spectral_density(eigenvalues, theta, spectral_form)
    
    # Find effective range in eigenvalue units
    lambda_eff = compute_effective_range(eigenvalues, p_lambda)
    
    # Convert to spatial range
    # For grid graphs: lambda ≈ 4 sin^2(π k / (2n)) where k is spatial frequency
    # Approximate spatial range as 1 / sqrt(lambda_eff)
    if lambda_eff > 0:
        spatial_range = grid_spacing / torch.sqrt(torch.tensor(lambda_eff))
    else:
        spatial_range = float('inf')
    
    return spatial_range.item()

def get_theta_dimension(spectral_form: str, poly_order: int = 5) -> int:
    """
    Get the dimension of theta parameter for a given spectral form.
    
    Args:
        spectral_form: Type of spectral density
        poly_order: Polynomial order (only used for 'chebyshev')
        
    Returns:
        Dimension of theta parameter vector
        
    Example:
        >>> get_theta_dimension('rational')
        2
        >>> get_theta_dimension('chebyshev', poly_order=5)
        6
    """
    if spectral_form == 'chebyshev':
        return poly_order + 1
    elif spectral_form in ['rational', 'exponential']:
        return 2
    elif spectral_form == 'power_law':
        return 3
    else:
        raise ValueError(f"Unknown spectral_form: {spectral_form}")


def interpret_theta_parameters(
    theta: torch.Tensor, 
    spectral_form: str
) -> dict:
    """
    Interpret learned theta parameters in terms of physical quantities.
    
    Args:
        theta: Spectral parameters (learned values, on log scale for parametric forms)
        spectral_form: Type of spectral density
        
    Returns:
        Dictionary with interpreted parameter values
        
    Example:
        >>> theta = torch.tensor([0.0, 1.0])
        >>> params = interpret_theta_parameters(theta, 'rational')
        >>> print(params)
        {'a': 1.0, 'b': 2.718, 'form': '1/(a + b*λ)'}
    """
    if spectral_form == 'rational':
        a = torch.exp(theta[0]).item()
        b = torch.exp(theta[1]).item()
        return {
            'a': a,
            'b': b,
            'form': '1/(a + b*λ)',
            'decay_rate': b / a
        }
    
    elif spectral_form == 'exponential':
        a = torch.exp(theta[0]).item()
        b = torch.exp(theta[1]).item()
        return {
            'a': a,
            'b': b,
            'form': 'exp(-a - b*λ)',
            'decay_rate': b
        }
    
    elif spectral_form == 'power_law':
        a = torch.exp(theta[0]).item()
        b = torch.exp(theta[1]).item()
        d = torch.exp(theta[2]).item()
        return {
            'a': a,
            'b': b,
            'd': d,
            'form': '1/(a + b*λ)^d',
            'decay_rate': b / a,
            'power': d
        }
    
    elif spectral_form == 'chebyshev':
        return {
            'coefficients': theta.detach().cpu().numpy(),
            'form': 'exp(Σ θ_k T_k(λ))'
        }
    
    else:
        raise ValueError(f"Unknown spectral_form: {spectral_form}")
