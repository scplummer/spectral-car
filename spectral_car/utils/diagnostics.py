"""
Diagnostic Utilities functions for spectral CAR models.

"""

from typing import Tuple, List, Union

import torch
import numpy as np

# ============================================================================
# 
# ============================================================================

def compute_effective_sample_size(
    samples: torch.Tensor,
    axis: int = 0
) -> torch.Tensor:
    """
    Compute effective sample size (ESS) for MCMC chains or variational samples.
    
    Estimates ESS based on autocorrelation, following Gelman et al. (2013).
    For variational inference, this helps assess the quality of MC estimates.
    
    Args:
        samples: Sample tensor (n_samples, ...)
        axis: Axis along which samples vary (default: 0)
        
    Returns:
        ess: Effective sample size for each parameter
        
    Example:
        >>> samples = model.sample_spatial_field(1000)
        >>> ess = compute_effective_sample_size(samples)
        >>> print(f"Mean ESS: {ess.mean():.1f}")
    """
    n_samples = samples.shape[axis]
    
    # Center samples
    samples_centered = samples - samples.mean(dim=axis, keepdim=True)
    
    # Compute autocorrelation
    def autocorr(x, lag):
        if lag == 0:
            return torch.var(x, dim=axis)
        else:
            x1 = x.narrow(axis, 0, n_samples - lag)
            x2 = x.narrow(axis, lag, n_samples - lag)
            return (x1 * x2).mean(dim=axis)
    
    # Compute autocorrelation up to some maximum lag
    max_lag = min(n_samples // 2, 100)
    var0 = autocorr(samples_centered, 0)
    
    rho = []
    for lag in range(1, max_lag):
        rho_lag = autocorr(samples_centered, lag) / (var0 + 1e-10)
        rho.append(rho_lag)
        
        # Stop when autocorrelation becomes negative
        if (rho_lag < 0).all():
            break
    
    if len(rho) == 0:
        return torch.full_like(var0, float(n_samples))
    
    rho = torch.stack(rho, dim=0)
    
    # ESS formula: n / (1 + 2 * sum(rho))
    ess = n_samples / (1 + 2 * rho.sum(dim=0))
    
    # Clamp to reasonable range
    ess = torch.clamp(ess, min=1.0, max=float(n_samples))
    
    return ess


def compute_waic(
    log_likelihoods: torch.Tensor,
    return_components: bool = False
) -> Union[float, Tuple[float, float, float]]:
    """
    Compute Watanabe-Akaike Information Criterion (WAIC).
    
    WAIC is a fully Bayesian approach to model comparison that uses
    the full posterior distribution. Lower values indicate better models.
    
    Args:
        log_likelihoods: Log likelihood samples (n_samples, n_obs)
        return_components: If True, return (waic, lppd, p_waic)
        
    Returns:
        waic: WAIC value (or tuple if return_components=True)
        
    Reference:
        Watanabe (2010), Gelman et al. (2013)
        
    Example:
        >>> log_liks = model.compute_log_likelihood(y, X, n_samples=1000)
        >>> waic = compute_waic(log_liks)
        >>> print(f"WAIC: {waic:.2f}")
    """
    # Log pointwise predictive density
    # lppd = sum_i log(1/S * sum_s p(y_i | theta_s))
    lppd = torch.logsumexp(log_likelihoods, dim=0) - np.log(log_likelihoods.shape[0])
    lppd = lppd.sum()
    
    # Effective number of parameters
    # p_waic = sum_i Var_s(log p(y_i | theta_s))
    p_waic = log_likelihoods.var(dim=0).sum()
    
    # WAIC = -2 * (lppd - p_waic)
    waic = -2 * (lppd - p_waic)
    
    if return_components:
        return waic.item(), lppd.item(), p_waic.item()
    return waic.item()


def compute_loo(
    log_likelihoods: torch.Tensor,
    return_diagnostics: bool = False
) -> Union[float, dict]:
    """
    Compute Leave-One-Out Cross-Validation using Pareto Smoothed Importance Sampling.
    
    Approximates exact LOO-CV using importance sampling, following
    Vehtari, Gelman, and Gabry (2017).
    
    Args:
        log_likelihoods: Log likelihood samples (n_samples, n_obs)
        return_diagnostics: If True, return dict with diagnostics
        
    Returns:
        loo: LOO-CV estimate (or dict if return_diagnostics=True)
        
    Note:
        If any Pareto k > 0.7, the estimate may be unreliable.
        
    Example:
        >>> log_liks = model.compute_log_likelihood(y, X, n_samples=1000)
        >>> loo_dict = compute_loo(log_liks, return_diagnostics=True)
        >>> print(f"LOO: {loo_dict['loo']:.2f}")
        >>> print(f"Bad k: {loo_dict['n_bad_k']}")
    """
    n_samples, n_obs = log_likelihoods.shape
    
    # Compute importance ratios
    # r_i^s = 1 / p(y_i | theta_s)
    log_ratios = -log_likelihoods
    
    # For each observation, fit Pareto distribution to tail
    # and compute smoothed importance weights
    pointwise_loo = torch.zeros(n_obs)
    pareto_k = torch.zeros(n_obs)
    
    for i in range(n_obs):
        log_r_i = log_ratios[:, i]
        
        # Fit Pareto to upper tail (top 20%)
        M = int(0.2 * n_samples)
        log_r_sorted, _ = torch.sort(log_r_i)
        log_r_tail = log_r_sorted[-M:]
        
        # Estimate Pareto k (shape parameter)
        # k = 1 - mean(log(r_tail)) / log(max(r_tail))
        log_max = log_r_tail.max()
        if log_max > log_r_tail.mean():
            k = 1.0 - (log_r_tail.mean() - log_r_sorted[-M].item()) / (log_max - log_r_sorted[-M].item() + 1e-10)
        else:
            k = torch.tensor(0.5)
        
        pareto_k[i] = k
        
        # If k < 0.7, use importance sampling
        if k < 0.7:
            # Standard importance sampling
            log_weights = log_r_i - torch.logsumexp(log_r_i, dim=0)
            weights = torch.exp(log_weights)
            
            # LOO for this observation
            log_lik_i = log_likelihoods[:, i]
            pointwise_loo[i] = torch.logsumexp(log_lik_i + log_weights, dim=0)
        else:
            # Fall back to standard importance sampling (may be unreliable)
            log_weights = log_r_i - torch.logsumexp(log_r_i, dim=0)
            log_lik_i = log_likelihoods[:, i]
            pointwise_loo[i] = torch.logsumexp(log_lik_i + log_weights, dim=0)
    
    # Total LOO
    loo = pointwise_loo.sum()
    
    # Effective number of parameters
    p_loo = (torch.logsumexp(log_likelihoods, dim=0) - np.log(n_samples) - pointwise_loo).sum()
    
    if return_diagnostics:
        return {
            'loo': -2 * loo.item(),
            'p_loo': p_loo.item(),
            'pointwise_loo': pointwise_loo,
            'pareto_k': pareto_k,
            'n_bad_k': (pareto_k > 0.7).sum().item(),
            'max_pareto_k': pareto_k.max().item()
        }
    
    return -2 * loo.item()


def compare_models(
    models_dict: dict,
    y: torch.Tensor,
    X: torch.Tensor,
    n_samples: int = 1000,
    criterion: str = 'waic'
) -> dict:
    """
    Compare multiple models using information criteria.
    
    Args:
        models_dict: Dictionary of {name: model} to compare
        y: Observations
        X: Covariates
        n_samples: Number of samples for computing IC
        criterion: 'waic' or 'loo'
        
    Returns:
        results: Dictionary with comparison results
        
    Example:
        >>> models = {
        ...     'meanfield': model1,
        ...     'lowrank': model2,
        ...     'collapsed': model3
        ... }
        >>> comparison = compare_models(models, y, X)
        >>> print(comparison['ranking'])
    """
    results = {}
    
    for name, model in models_dict.items():
        # Get log likelihoods
        log_liks = model.compute_log_likelihood(y, X, n_samples=n_samples)
        
        # Compute criterion
        if criterion == 'waic':
            ic, lppd, p_ic = compute_waic(log_liks, return_components=True)
            results[name] = {
                'ic': ic,
                'lppd': lppd,
                'p_ic': p_ic
            }
        elif criterion == 'loo':
            loo_dict = compute_loo(log_liks, return_diagnostics=True)
            results[name] = loo_dict
            results[name]['ic'] = loo_dict['loo']
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    # Rank models
    ranking = sorted(results.items(), key=lambda x: x[1]['ic'])
    
    # Compute differences from best model
    best_ic = ranking[0][1]['ic']
    for name in results:
        results[name]['delta_ic'] = results[name]['ic'] - best_ic
    
    return {
        'results': results,
        'ranking': [(name, r['ic']) for name, r in ranking],
        'best_model': ranking[0][0],
        'criterion': criterion
    }


def check_convergence_diagnostics(
    history: List[dict],
    window: int = 100,
    tol: float = 0.1
) -> dict:
    """
    Check convergence diagnostics from training history.
    
    Args:
        history: List of diagnostic dictionaries from training
        window: Window size for computing recent improvement
        tol: Tolerance for convergence (relative ELBO change)
        
    Returns:
        diagnostics: Dictionary of convergence diagnostics
        
    Example:
        >>> diagnostics = check_convergence_diagnostics(
        ...     model.training_history,
        ...     window=100,
        ...     tol=0.001
        ... )
        >>> if diagnostics['converged']:
        ...     print("Model has converged!")
    """
    if len(history) < 2:
        return {'converged': False, 'message': 'Insufficient history'}
    
    elbos = torch.tensor([h['elbo'] for h in history])
    
    # Overall improvement
    total_improvement = elbos[-1] - elbos[0]
    
    # Recent improvement (over window)
    if len(elbos) >= window:
        recent_improvement = elbos[-1] - elbos[-window]
        relative_improvement = abs(recent_improvement / (abs(elbos[-window]) + 1e-10))
    else:
        recent_improvement = total_improvement
        relative_improvement = abs(total_improvement / (abs(elbos[0]) + 1e-10))
    
    # Check for convergence
    converged = relative_improvement < tol
    
    # Check for monotonicity (mostly increasing)
    elbo_diffs = elbos[1:] - elbos[:-1]
    n_decreases = (elbo_diffs < 0).sum().item()
    monotonic_ratio = 1.0 - n_decreases / len(elbo_diffs)
    
    # Compute coefficient of variation in recent window
    if len(elbos) >= window:
        recent_elbos = elbos[-window:]
        cv = recent_elbos.std() / (abs(recent_elbos.mean()) + 1e-10)
    else:
        cv = elbos.std() / (abs(elbos.mean()) + 1e-10)
    
    return {
        'converged': converged,
        'total_improvement': total_improvement.item(),
        'recent_improvement': recent_improvement.item(),
        'relative_improvement': relative_improvement.item(),
        'n_iterations': len(elbos),
        'monotonic_ratio': monotonic_ratio,
        'coefficient_variation': cv.item(),
        'final_elbo': elbos[-1].item(),
        'message': 'Converged' if converged else 'Not yet converged'
    }


def compute_prediction_intervals(
    samples: torch.Tensor,
    coverage: float = 0.95,
    method: str = 'quantile'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute prediction intervals from posterior samples.
    
    Args:
        samples: Posterior samples (n_samples, n_obs)
        coverage: Target coverage probability (default: 0.95)
        method: 'quantile' or 'hpd' (highest posterior density)
        
    Returns:
        lower: Lower bounds (n_obs,)
        upper: Upper bounds (n_obs,)
        
    Example:
        >>> samples = model.sample_posterior_predictive(y, X, n_samples=1000)
        >>> lower, upper = compute_prediction_intervals(samples, coverage=0.95)
        >>> in_interval = ((y >= lower) & (y <= upper)).float().mean()
        >>> print(f"Empirical coverage: {in_interval:.3f}")
    """
    alpha = 1 - coverage
    
    if method == 'quantile':
        # Equal-tailed interval
        lower = torch.quantile(samples, alpha / 2, dim=0)
        upper = torch.quantile(samples, 1 - alpha / 2, dim=0)
        
    elif method == 'hpd':
        # Highest posterior density interval
        # Sort samples
        samples_sorted, _ = torch.sort(samples, dim=0)
        n_samples = samples.shape[0]
        
        # Interval width for desired coverage
        interval_size = int(coverage * n_samples)
        
        # Find shortest interval
        n_obs = samples.shape[1]
        lower = torch.zeros(n_obs)
        upper = torch.zeros(n_obs)
        
        for i in range(n_obs):
            # Compute all possible interval widths
            widths = samples_sorted[interval_size:, i] - samples_sorted[:n_samples-interval_size, i]
            
            # Find shortest interval
            min_idx = widths.argmin()
            lower[i] = samples_sorted[min_idx, i]
            upper[i] = samples_sorted[min_idx + interval_size, i]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return lower, upper


def compute_coverage(
    predictions: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    true_values: torch.Tensor
) -> dict:
    """
    Compute empirical coverage statistics.
    
    Args:
        predictions: Point predictions (n_obs,)
        lower: Lower interval bounds (n_obs,)
        upper: Upper interval bounds (n_obs,)
        true_values: True values (n_obs,)
        
    Returns:
        stats: Dictionary of coverage statistics
        
    Example:
        >>> lower, upper = compute_prediction_intervals(samples, 0.95)
        >>> stats = compute_coverage(predictions, lower, upper, y_true)
        >>> print(f"Coverage: {stats['coverage']:.3f}")
    """
    # Check coverage
    in_interval = ((true_values >= lower) & (true_values <= upper)).float()
    coverage = in_interval.mean().item()
    
    # Interval width
    width = (upper - lower).mean().item()
    
    # Prediction error
    errors = (predictions - true_values).abs()
    mae = errors.mean().item()
    rmse = (errors ** 2).mean().sqrt().item()
    
    # Check miscoverage direction
    below = (true_values < lower).float().mean().item()
    above = (true_values > upper).float().mean().item()
    
    return {
        'coverage': coverage,
        'mean_interval_width': width,
        'mae': mae,
        'rmse': rmse,
        'below_rate': below,
        'above_rate': above,
        'n_obs': len(true_values)
    }


def compute_spatial_autocorrelation(
    residuals: torch.Tensor,
    W: torch.Tensor
) -> float:
    """
    Compute Moran's I statistic for spatial autocorrelation in residuals.
    
    Moran's I measures spatial autocorrelation. Values near 0 indicate
    no spatial correlation (good for model residuals).
    
    Args:
        residuals: Model residuals (n_obs,)
        W: Spatial weights matrix (n_obs, n_obs)
        
    Returns:
        morans_i: Moran's I statistic
        
    Reference:
        Moran (1950), Cliff and Ord (1981)
        
    Example:
        >>> residuals = y - predictions
        >>> W = create_adjacency_matrix(coords, 'knn', k=5)
        >>> morans_i = compute_spatial_autocorrelation(residuals, W)
        >>> print(f"Moran's I: {morans_i:.3f}")
    """
    n = len(residuals)
    
    # Center residuals
    r_centered = residuals - residuals.mean()
    
    # Numerator: sum_i sum_j W_ij * r_i * r_j
    numerator = (W * torch.outer(r_centered, r_centered)).sum()
    
    # Denominator: sum_i r_i^2
    denominator = (r_centered ** 2).sum()
    
    # Total weight
    W_sum = W.sum()
    
    # Moran's I
    morans_i = (n / W_sum) * (numerator / denominator)
    
    return morans_i.item()