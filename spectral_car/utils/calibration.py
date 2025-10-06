"""
Calibration Utilities functions for spectral CAR models.
"""

from typing import Tuple, List

import torch
import numpy as np
from scipy import stats


def find_calibration_factor(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    true_values: torch.Tensor,
    target_coverage: float = 0.95,
    z_score: float = 1.96,
    method: str = 'binary_search'
) -> float:
    """
    Find calibration factor for uncertainties to achieve target coverage.
    
    Searches for factor f such that:
    P(|true - pred| < f * z * std) = target_coverage
    
    This is the core calibration function used by CalibratedVI.
    
    Args:
        predictions: Predicted values (n,)
        uncertainties: Predicted standard deviations (n,)
        true_values: True values (n,)
        target_coverage: Desired coverage probability (e.g., 0.95)
        z_score: Z-score for confidence intervals (1.96 for 95%, 1.0 for 68%)
        method: Calibration method ('binary_search' or 'quantile')
        
    Returns:
        Calibration factor (scalar) to multiply uncertainties by
        
    Example:
        >>> pred = torch.randn(100)
        >>> std = torch.ones(100) * 0.5
        >>> true = pred + torch.randn(100) * 0.3
        >>> factor = find_calibration_factor(pred, std, true, target_coverage=0.95)
        >>> print(f"Scale uncertainties by {factor:.2f}x")
    """
    errors = torch.abs(true_values - predictions)
    
    if method == 'binary_search':
        def coverage_at_factor(f):
            intervals = f * z_score * uncertainties
            return torch.mean((errors <= intervals).float()).item()
        
        # Binary search
        f_low, f_high = 0.01, 5.0
        tolerance = 0.001
        max_iterations = 100
        
        for _ in range(max_iterations):
            if f_high - f_low < tolerance:
                break
                
            f_mid = (f_low + f_high) / 2
            cov = coverage_at_factor(f_mid)
            
            if cov < target_coverage:
                f_low = f_mid
            else:
                f_high = f_mid
        
        return (f_low + f_high) / 2
    
    elif method == 'quantile':
        # Use quantile of standardized errors
        standardized_errors = errors / (uncertainties + 1e-8)
        quantile = torch.quantile(standardized_errors, target_coverage).item()
        return quantile / z_score
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def compute_coverage(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    true_values: torch.Tensor,
    confidence_levels: List[float] = [0.68, 0.95, 0.99]
) -> dict:
    """
    Compute empirical coverage at multiple confidence levels.
    
    Args:
        predictions: Predicted values (n,)
        uncertainties: Predicted standard deviations (n,)
        true_values: True values (n,)
        confidence_levels: List of confidence levels to check
        
    Returns:
        Dictionary mapping confidence level to empirical coverage
        
    Example:
        >>> coverage = compute_coverage(pred, std, true, [0.68, 0.95])
        >>> print(f"68% interval coverage: {coverage[0.68]:.1%}")
    """
    errors = torch.abs(true_values - predictions)
    coverages = {}
    
    for conf in confidence_levels:
        # Get z-score for this confidence level
        z = confidence_to_z_score(conf)
        
        # Compute coverage
        intervals = z * uncertainties
        coverage = torch.mean((errors <= intervals).float()).item()
        coverages[conf] = coverage
    
    return coverages


def confidence_to_z_score(confidence: float) -> float:
    """
    Convert confidence level to z-score.
    
    Args:
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Z-score (e.g., 1.96 for 95%)
        
    Example:
        >>> z = confidence_to_z_score(0.95)
        >>> print(z)  # 1.96
    """
    
    return float(stats.norm.ppf((1 + confidence) / 2))
  

def z_score_to_confidence(z: float) -> float:
    """
    Convert z-score to confidence level.
    
    Args:
        z: Z-score (e.g., 1.96)
        
    Returns:
        Confidence level (e.g., 0.95)
    """

    return float(2 * stats.norm.cdf(z) - 1)

def calibration_curve(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    true_values: torch.Tensor,
    n_bins: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute calibration curve (reliability diagram).
    
    Bins predictions by uncertainty level and computes empirical coverage
    in each bin. Well-calibrated predictions should have coverage equal to
    the predicted confidence level.
    
    Args:
        predictions: Predicted values (n,)
        uncertainties: Predicted standard deviations (n,)
        true_values: True values (n,)
        n_bins: Number of bins for uncertainty levels
        
    Returns:
        predicted_confidence: Predicted confidence levels (n_bins,)
        empirical_coverage: Empirical coverage in each bin (n_bins,)
        
    Example:
        >>> pred_conf, emp_cov = calibration_curve(pred, std, true)
        >>> plt.plot(pred_conf, emp_cov, label='Calibration curve')
        >>> plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    """
    # Compute standardized errors
    errors = torch.abs(true_values - predictions)
    standardized_errors = errors / (uncertainties + 1e-8)
    
    # Bin by uncertainty level
    uncertainty_quantiles = torch.linspace(0, 1, n_bins + 1)
    uncertainty_bins = torch.quantile(uncertainties, uncertainty_quantiles)
    
    predicted_confidence = []
    empirical_coverage = []
    
    for i in range(n_bins):
        # Find samples in this uncertainty bin
        in_bin = (uncertainties >= uncertainty_bins[i]) & (uncertainties < uncertainty_bins[i + 1])
        if i == n_bins - 1:  # Include upper bound in last bin
            in_bin = in_bin | (uncertainties == uncertainty_bins[i + 1])
        
        if torch.sum(in_bin) == 0:
            continue
        
        # Average uncertainty in this bin
        avg_uncertainty = torch.mean(uncertainties[in_bin]).item()
        
        # Convert to predicted confidence (assuming Gaussian)
        # For 1-sigma interval: confidence ≈ 0.68
        pred_conf = z_score_to_confidence(1.0)  # Simplified
        
        # Empirical coverage in this bin
        emp_cov = torch.mean((standardized_errors[in_bin] <= 1.0).float()).item()
        
        predicted_confidence.append(pred_conf)
        empirical_coverage.append(emp_cov)
    
    return torch.tensor(predicted_confidence), torch.tensor(empirical_coverage)


def sharpness(uncertainties: torch.Tensor) -> float:
    """
    Compute sharpness (average uncertainty).
    
    Sharpness measures how confident predictions are. Lower is better
    (more confident), but must be balanced with calibration.
    
    Args:
        uncertainties: Predicted standard deviations (n,)
        
    Returns:
        Average uncertainty (sharpness)
    """
    return torch.mean(uncertainties).item()


def continuous_ranked_probability_score(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    true_values: torch.Tensor
) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS).
    
    CRPS is a proper scoring rule that evaluates both calibration and sharpness.
    Lower is better.
    
    For Gaussian predictions: CRPS = σ * [z/√π - 2φ(z) - 1/√π]
    where z = (true - pred) / σ, φ is standard normal CDF
    
    Args:
        predictions: Predicted means (n,)
        uncertainties: Predicted standard deviations (n,)
        true_values: True values (n,)
        
    Returns:
        Average CRPS across all predictions
        
    Example:
        >>> crps = continuous_ranked_probability_score(pred, std, true)
        >>> print(f"CRPS: {crps:.3f}")  # Lower is better
    """
    # Standardized errors
    z = (true_values - predictions) / (uncertainties + 1e-8)
    
    # CRPS for Gaussian distribution
    # CRPS = σ * [z/√π - 2φ(z) - 1/√π]
    sqrt_pi = np.sqrt(np.pi)
    
    # Standard normal CDF (approximate if scipy not available)
    phi_z = torch.tensor([stats.norm.cdf(z_i.item()) for z_i in z])
    
    crps = uncertainties * (z / sqrt_pi - 2 * phi_z - 1 / sqrt_pi)
    
    return torch.mean(crps).item()


def interval_score(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    true_values: torch.Tensor,
    confidence: float = 0.95
) -> float:
    """
    Compute interval score (proper scoring rule).
    
    Interval score penalizes both interval width and miscalibration.
    Lower is better.
    
    Args:
        predictions: Predicted means (n,)
        uncertainties: Predicted standard deviations (n,)
        true_values: True values (n,)
        confidence: Confidence level for intervals (e.g., 0.95)
        
    Returns:
        Average interval score
    """
    alpha = 1 - confidence
    z = confidence_to_z_score(confidence)
    
    # Interval bounds
    lower = predictions - z * uncertainties
    upper = predictions + z * uncertainties
    
    # Interval score
    score = (upper - lower) + \
            (2 / alpha) * (lower - true_values) * (true_values < lower).float() + \
            (2 / alpha) * (true_values - upper) * (true_values > upper).float()
    
    return torch.mean(score).item()


def calibration_test(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    true_values: torch.Tensor,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> dict:
    """
    Statistical test for calibration using bootstrap.
    
    Tests null hypothesis: empirical coverage = target coverage
    
    Args:
        predictions: Predicted means (n,)
        uncertainties: Predicted standard deviations (n,)
        true_values: True values (n,)
        confidence: Target confidence level
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with test results including p-value
    """
    # Observed coverage
    z = confidence_to_z_score(confidence)
    errors = torch.abs(true_values - predictions)
    intervals = z * uncertainties
    observed_coverage = torch.mean((errors <= intervals).float()).item()
    
    # Bootstrap distribution under null hypothesis
    n = len(predictions)
    bootstrap_coverages = []
    
    for _ in range(n_bootstrap):
        # Resample
        indices = torch.randint(0, n, (n,))
        boot_errors = errors[indices]
        boot_intervals = intervals[indices]
        boot_coverage = torch.mean((boot_errors <= boot_intervals).float()).item()
        bootstrap_coverages.append(boot_coverage)
    
    bootstrap_coverages = np.array(bootstrap_coverages)
    
    # P-value (two-tailed)
    p_value = np.mean(np.abs(bootstrap_coverages - confidence) >= 
                     np.abs(observed_coverage - confidence))
    
    return {
        'observed_coverage': observed_coverage,
        'target_coverage': confidence,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'coverage_std': np.std(bootstrap_coverages)
    }