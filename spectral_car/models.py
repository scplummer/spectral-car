"""
Spectral CAR model definitions.

This module implements Conditional Autoregressive (CAR) spatial models
using spectral polynomial priors and variational inference.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class SpectralCARBase(nn.Module):
    """
    Base class for spectral CAR models.
    
    This abstract base class provides common functionality for spectral CAR models
    including eigendecomposition storage, Chebyshev polynomial computation, and
    spectral density evaluation.
    
    Args:
        n_obs: Number of spatial observations
        n_features: Number of fixed effect covariates
        eigenvalues: Eigenvalues of graph Laplacian (n_obs,)
        eigenvectors: Eigenvectors of graph Laplacian (n_obs, n_obs)
        poly_order: Order of Chebyshev polynomial for spectral filter
        prior_beta_mean: Prior mean for fixed effects (default: zeros)
        prior_beta_std: Prior std for fixed effects (default: 10.0)
        prior_theta_mean: Prior mean for spectral coefficients (default: zeros)
        prior_theta_std: Prior std for spectral coefficients (default: 1.0)
        prior_tau_a: Prior shape for tau^2 (spatial variance)
        prior_tau_b: Prior rate for tau^2
        prior_sigma_a: Prior shape for sigma^2 (observation noise)
        prior_sigma_b: Prior rate for sigma^2
    """
    
    def __init__(
        self,
        n_obs: int,
        n_features: int,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        poly_order: int = 5,
        prior_beta_mean: Optional[torch.Tensor] = None,
        prior_beta_std: float = 5.0,
        prior_theta_mean: Optional[torch.Tensor] = None,
        prior_theta_std: float = 1.0,
        prior_tau_a: float = 3.0,
        prior_tau_b: float = 1.0,
        prior_sigma_a: float = 5.0,
        prior_sigma_b: float = 1.0,
    ):
        super().__init__()
        
        self.n_obs = n_obs
        self.n_features = n_features
        self.poly_order = poly_order
        
        # Store eigendecomposition as buffers (not parameters)
        self.register_buffer('eigenvalues', eigenvalues)
        self.register_buffer('eigenvectors', eigenvectors)
        
        # Normalize eigenvalues to [-1, 1] for Chebyshev stability
        self._normalize_eigenvalues()
        
        # Store priors as buffers
        self.register_buffer('prior_beta_mean', 
                           prior_beta_mean if prior_beta_mean is not None 
                           else torch.zeros(n_features))
        self.register_buffer('prior_beta_cov', 
                           torch.eye(n_features) * prior_beta_std**2)
        
        self.register_buffer('prior_theta_mean',
                           prior_theta_mean if prior_theta_mean is not None
                           else torch.zeros(poly_order + 1))
        self.register_buffer('prior_theta_cov',
                           torch.eye(poly_order + 1) * prior_theta_std**2)
        
        self.prior_tau_a = prior_tau_a
        self.prior_tau_b = prior_tau_b
        self.prior_sigma_a = prior_sigma_a
        self.prior_sigma_b = prior_sigma_b
        
    def _normalize_eigenvalues(self):
        """Normalize eigenvalues to [-1, 1] for Chebyshev polynomial stability."""
        lambda_min = self.eigenvalues.min()
        lambda_max = self.eigenvalues.max()
        self.register_buffer('lambda_min', lambda_min)
        self.register_buffer('lambda_max', lambda_max)
        self.register_buffer(
            'eigenvalues_normalized',
            2 * (self.eigenvalues - lambda_min) / (lambda_max - lambda_min + 1e-8) - 1
        )
    
    def chebyshev_polynomials(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Chebyshev polynomials T_0(x), ..., T_K(x).
        
        Uses the recurrence relation: T_{k+1}(x) = 2x*T_k(x) - T_{k-1}(x)
        
        Args:
            x: Input values, shape (n,)
            
        Returns:
            Tensor of shape (n, K+1) where [:, k] contains T_k(x)
        """
        n = x.shape[0]
        T = torch.zeros(n, self.poly_order + 1, device=x.device)
        T[:, 0] = 1.0
        if self.poly_order >= 1:
            T[:, 1] = x
        for k in range(2, self.poly_order + 1):
            T[:, k] = 2 * x * T[:, k-1] - T[:, k-2]
        return T
    
    def spectral_density(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral density p(lambda_j; theta) = exp(sum_k theta_k * T_k(lambda_j)).
        
        This defines the precision at each spatial frequency (eigenvalue).
        
        Args:
            theta: Polynomial coefficients, shape (K+1,) or (batch, K+1)
            
        Returns:
            Spectral density values, shape (n_obs,) or (batch, n_obs)
        """
        T = self.chebyshev_polynomials(self.eigenvalues_normalized)  # (n_obs, K+1)
        
        if theta.dim() == 1:
            # Single sample: T @ theta
            log_p = torch.matmul(T, theta)  # (n_obs,)
        else:
            # Batch of samples: T @ theta.T -> (n_obs, batch) -> transpose to (batch, n_obs)
            log_p = torch.matmul(T, theta.T).T  # (batch, n_obs)
        
        spectral_density = torch.exp(log_p)
        spectral_density = torch.clamp(spectral_density, min=1e-6, max=1e6)
        return spectral_density
    
    def get_parameter_summary(self) -> dict:
        """
        Get summary of estimated parameters.
        
        Returns:
            Dictionary with parameter means and standard deviations
        """
        with torch.no_grad():
            # Variance parameters - check if a > 2 for valid variance
            tau2_mean = (self.b_tau / (self.a_tau - 1)).item()
            tau2_var = self.b_tau**2 / ((self.a_tau - 1)**2 * torch.clamp(self.a_tau - 2, min=1e-6))
            tau2_std = torch.sqrt(tau2_var).item()
            
            sigma2_mean = (self.b_sigma / (self.a_sigma - 1)).item()
            sigma2_var = self.b_sigma**2 / ((self.a_sigma - 1)**2 * torch.clamp(self.a_sigma - 2, min=1e-6))
            sigma2_std = torch.sqrt(sigma2_var).item()
            
            summary = {
                'beta_mean': self.mu_beta.cpu().numpy(),
                'beta_std': self.sigma_beta.cpu().numpy(),
                'theta_mean': self.mu_theta.cpu().numpy(),
                'theta_std': self.sigma_theta.cpu().numpy(),
                'tau2_mean': tau2_mean,
                'tau2_std': tau2_std,
                'sigma2_mean': sigma2_mean,
                'sigma2_std': sigma2_std,
            }
            
            return summary


class SpectralCARMeanField(SpectralCARBase):
    """
    Spectral CAR model with mean-field variational inference (marginalized).
    
    This is the computationally efficient version that marginalizes out the
    spatial random effects phi analytically. Best for hyperparameter estimation
    when you don't need uncertainty quantification for individual spatial effects.
    
    Variational family:
        q(beta, theta, tau^2, sigma^2) = q(beta)q(theta)q(tau^2)q(sigma^2)
    
    Spatial effects phi are marginalized out, leading to O(n) per iteration.
    
    Args:
        (inherits all args from SpectralCARBase)
        n_mc_samples: Number of Monte Carlo samples for ELBO approximation
    """
    
    def __init__(
        self,
        n_obs: int,
        n_features: int,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        poly_order: int = 5,
        n_mc_samples: int = 20,
        **kwargs
    ):
        super().__init__(n_obs, n_features, eigenvalues, eigenvectors, poly_order, **kwargs)
        
        self.n_mc_samples_initial = n_mc_samples
        self.n_mc_samples = n_mc_samples
        
        self._init_variational_parameters()
        
    def _init_variational_parameters(self):
        """Initialize variational distribution parameters."""
        # q(beta) = N(mu_beta, diag(sigma_beta^2))
        self.mu_beta = nn.Parameter(torch.zeros(self.n_features))
        self.log_diag_sigma_beta = nn.Parameter(torch.zeros(self.n_features) - 0.5)
        
        # q(theta) = N(mu_theta, diag(sigma_theta^2))
        init_theta = torch.zeros(self.poly_order + 1)
        init_theta[0] = 0.5  # Bias towards positive spectral density
        self.mu_theta = nn.Parameter(init_theta)
        self.log_diag_sigma_theta = nn.Parameter(torch.ones(self.poly_order + 1) * (-2))
        
        # q(tau^2) = InverseGamma(a_tau, b_tau)
        init_a_tau = 5.0
        init_b_tau = (init_a_tau - 1) * 0.5
        self.log_a_tau = nn.Parameter(torch.log(torch.tensor(init_a_tau)))
        self.log_b_tau = nn.Parameter(torch.log(torch.tensor(init_b_tau)))
        
        # q(sigma^2) = InverseGamma(a_sigma, b_sigma)
        init_a_sigma = 6.0
        init_b_sigma = (init_a_sigma - 1) * 0.25
        self.log_a_sigma = nn.Parameter(torch.log(torch.tensor(init_a_sigma)))
        self.log_b_sigma = nn.Parameter(torch.log(torch.tensor(init_b_sigma)))
    
    @property
    def sigma_beta(self) -> torch.Tensor:
        """Standard deviations for beta."""
        return torch.exp(self.log_diag_sigma_beta)
    
    @property
    def sigma_theta(self) -> torch.Tensor:
        """Standard deviations for theta."""
        return torch.exp(self.log_diag_sigma_theta)
    
    @property
    def a_tau(self) -> torch.Tensor:
        """Shape parameter for tau^2."""
        return torch.exp(self.log_a_tau)
    
    @property
    def b_tau(self) -> torch.Tensor:
        """Rate parameter for tau^2."""
        return torch.exp(self.log_b_tau)
    
    @property
    def a_sigma(self) -> torch.Tensor:
        """Shape parameter for sigma^2."""
        return torch.exp(self.log_a_sigma)
    
    @property
    def b_sigma(self) -> torch.Tensor:
        """Rate parameter for sigma^2."""
        return torch.exp(self.log_b_sigma)
    
    def sample_variational_params(self, n_samples: int) -> dict:
        """
        Sample from variational distributions.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Dictionary with samples for each parameter
        """
        device = self.mu_beta.device

        # Sample beta
        beta_samples = self.mu_beta + self.sigma_beta * torch.randn(
            n_samples, self.n_features, device=device
        )
        
        # Sample theta
        theta_samples = self.mu_theta + self.sigma_theta * torch.randn(
            n_samples, self.poly_order + 1, device=device
        )
        
        # Sample tau^2 using Inverse Gamma: sample via 1/Gamma(a, 1/b)
        gamma_samples = torch.distributions.Gamma(
            self.a_tau, 1.0 / self.b_tau
        ).sample((n_samples,))
        tau2_samples = 1.0 / gamma_samples
        
        # Sample sigma^2
        gamma_samples = torch.distributions.Gamma(
            self.a_sigma, 1.0 / self.b_sigma
        ).sample((n_samples,))
        sigma2_samples = 1.0 / gamma_samples
        
        return {
            'beta': beta_samples,
            'theta': theta_samples,
            'tau2': tau2_samples,
            'sigma2': sigma2_samples
        }
    
    def marginal_log_likelihood(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        beta: torch.Tensor,
        theta: torch.Tensor,
        tau2: torch.Tensor,
        sigma2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log p(y | beta, theta, tau^2, sigma^2) with phi marginalized.
        
        Uses spectral decomposition for O(n) computation.
        
        Args:
            y: Observations (n_obs,)
            X: Design matrix (n_obs, n_features)
            beta: Fixed effects, shape (n_features,) or (batch, n_features)
            theta: Spectral coefficients, shape (K+1,) or (batch, K+1)
            tau2: Spatial variance, shape () or (batch,)
            sigma2: Observation variance, shape () or (batch,)
            
        Returns:
            Log likelihood value (scalar or batch)
        """
        # Compute spectral density p(lambda_j; theta)
        p_lambda = self.spectral_density(theta)  # (n_obs,) or (batch, n_obs)
        
        # Add small epsilon for numerical stability
        eps = 1e-6
        p_lambda = torch.clamp(p_lambda, min=eps)
        
        if theta.dim() == 1:
            # Single sample
            denom = sigma2 * tau2 * p_lambda + 1.0
            D_diag = tau2 * p_lambda / denom
            
            # Compute residuals
            residual = y - X @ beta  # (n_obs,)
            
            # Transform to spectral domain
            y_tilde = self.eigenvectors.T @ residual  # (n_obs,)
            
            # Log determinant
            log_det = (torch.sum(torch.log(denom)) - self.n_obs * torch.log(tau2) 
                      - torch.sum(torch.log(p_lambda)))
            
            # Quadratic form
            quad_form = torch.sum(D_diag * y_tilde**2)
            
        else:
            # Batch of samples
            batch_size = theta.shape[0]
            p_lambda = p_lambda.view(batch_size, self.n_obs)
            tau2 = tau2.view(batch_size, 1)
            sigma2 = sigma2.view(batch_size, 1)
            
            denom = sigma2 * tau2 * p_lambda + 1.0
            D_diag = tau2 * p_lambda / denom
            
            # Compute residuals: y - X @ beta
            Xbeta = torch.matmul(beta, X.T)  # (batch, n_obs)
            residual = y.unsqueeze(0) - Xbeta  # (batch, n_obs)
            
            # Transform to spectral domain
            y_tilde = torch.matmul(residual, self.eigenvectors)  # (batch, n_obs)
            
            log_det = (torch.sum(torch.log(denom), dim=1) 
                      - self.n_obs * torch.log(tau2.squeeze()) 
                      - torch.sum(torch.log(p_lambda), dim=1))
            
            quad_form = torch.sum(D_diag * y_tilde**2, dim=1)
        
        log_lik = -0.5 * self.n_obs * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad_form
        return log_lik
    
    def kl_divergence_terms(self) -> dict:
        """
        Compute KL divergences between variational and prior distributions.
        
        Returns:
            Dictionary of KL divergence values for each parameter
        """
        # KL(q(beta) || p(beta))
        kl_beta = 0.5 * (
            torch.sum(self.sigma_beta**2 / torch.diag(self.prior_beta_cov))
            + torch.sum(((self.mu_beta - self.prior_beta_mean)**2) / torch.diag(self.prior_beta_cov))
            - self.n_features
            + torch.sum(torch.log(torch.diag(self.prior_beta_cov)))
            - 2 * torch.sum(self.log_diag_sigma_beta)
        )
        
        # KL(q(theta) || p(theta))
        kl_theta = 0.5 * (
            torch.sum(self.sigma_theta**2 / torch.diag(self.prior_theta_cov))
            + torch.sum(((self.mu_theta - self.prior_theta_mean)**2) / torch.diag(self.prior_theta_cov))
            - (self.poly_order + 1)
            + torch.sum(torch.log(torch.diag(self.prior_theta_cov)))
            - 2 * torch.sum(self.log_diag_sigma_theta)
        )
        
        # KL(q(tau^2) || p(tau^2)) for InverseGamma
        kl_tau = (
            self.a_tau * torch.log(self.b_tau / self.prior_tau_b)
            - torch.lgamma(self.a_tau) + torch.lgamma(torch.tensor(self.prior_tau_a))
            + (self.prior_tau_a - self.a_tau) * (torch.log(self.b_tau) - torch.digamma(self.a_tau))
            + self.a_tau * (self.prior_tau_b / self.b_tau - 1.0)
        )
        
        # KL(q(sigma^2) || p(sigma^2))
        kl_sigma = (
            self.a_sigma * torch.log(self.b_sigma / self.prior_sigma_b)
            - torch.lgamma(self.a_sigma) + torch.lgamma(torch.tensor(self.prior_sigma_a))
            + (self.prior_sigma_a - self.a_sigma) * (torch.log(self.b_sigma) - torch.digamma(self.a_sigma))
            + self.a_sigma * (self.prior_sigma_b / self.b_sigma - 1.0)
        )
        
        return {
            'kl_beta': kl_beta,
            'kl_theta': kl_theta,
            'kl_tau': kl_tau,
            'kl_sigma': kl_sigma
        }
    
    def elbo(self, y: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute the Evidence Lower Bound (ELBO) using MC approximation.
        
        Args:
            y: Observations (n_obs,)
            X: Design matrix (n_obs, n_features)
            
        Returns:
            elbo_value: ELBO (scalar, to be maximized)
            diagnostics: Dictionary of diagnostic values
        """
        # Sample from variational distributions
        samples = self.sample_variational_params(self.n_mc_samples)
        
        # E[log p(y | params)] via MC
        log_liks = self.marginal_log_likelihood(
            y, X, 
            samples['beta'], 
            samples['theta'],
            samples['tau2'],
            samples['sigma2']
        )
        expected_log_lik = torch.mean(log_liks)
        
        # KL divergences (analytical)
        kl_terms = self.kl_divergence_terms()
        total_kl = sum(kl_terms.values())
        
        # ELBO = E[log p(y|params)] - KL
        elbo_value = expected_log_lik - total_kl
        
        diagnostics = {
            'expected_log_lik': expected_log_lik.item(),
            'total_kl': total_kl.item(),
            'log_lik_std': torch.std(log_liks).item(),
            **{k: v.item() for k, v in kl_terms.items()}
        }
        
        return elbo_value, diagnostics
    
    def predict_spatial_effect(self, y: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the spatial random effect phi given data (post-hoc prediction).
        
        Args:
            y: Observations (n_obs,)
            X: Design matrix (n_obs, n_features)
            
        Returns:
            phi_mean: Posterior mean of spatial effects (n_obs,)
            phi_std: Posterior std of spatial effects (n_obs,)
        """
        with torch.no_grad():
            # Use posterior mean of hyperparameters
            beta = self.mu_beta
            theta = self.mu_theta
            tau2 = self.b_tau / (self.a_tau - 1)  # E[tau^2]
            sigma2 = self.b_sigma / (self.a_sigma - 1)  # E[sigma^2]
            
            # Compute residuals
            residual = y - X @ beta
            
            # Transform to spectral domain
            y_tilde = self.eigenvectors.T @ residual
            
            # Compute spectral density
            p_lambda = self.spectral_density(theta)
            p_lambda = torch.clamp(p_lambda, min=1e-6)
            
            # Posterior mean in spectral domain: M_jj * y_tilde_j
            # M_jj = 1 / (sigma^2 * tau^2 * p_j + 1)
            M_diag = 1.0 / (sigma2 * tau2 * p_lambda + 1.0)
            alpha_mean = M_diag * y_tilde
            
            # Transform back to spatial domain
            phi_mean = self.eigenvectors @ alpha_mean
            
            # Posterior variance (diagonal approximation)
            phi_var = self.eigenvectors @ (M_diag.unsqueeze(-1) * self.eigenvectors.T)
            phi_std = torch.sqrt(torch.diag(phi_var))
            
            return phi_mean, phi_std


class SpectralCARJoint(SpectralCARBase):
    """
    Spectral CAR model with joint variational inference over hyperparameters AND spatial effects.
    
    This version explicitly includes spatial effects phi in the variational family,
    providing direct uncertainty quantification for spatial predictions.
    
    Variational family:
        q(beta, alpha, theta, tau^2, sigma^2) = q(beta)q(alpha)q(theta)q(tau^2)q(sigma^2)
    
    Where phi = U @ alpha (spectral parameterization).
    
    Args:
        (inherits all args from SpectralCARBase)
        n_mc_samples: Number of Monte Carlo samples for ELBO approximation
        center_spatial: Whether to enforce mean(phi) = 0 constraint
    """
    
    def __init__(
        self,
        n_obs: int,
        n_features: int,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        poly_order: int = 5,
        n_mc_samples: int = 20,
        center_spatial: bool = True,
        **kwargs
    ):
        super().__init__(n_obs, n_features, eigenvalues, eigenvectors, poly_order, **kwargs)
        
        self.n_mc_samples_initial = n_mc_samples
        self.n_mc_samples = n_mc_samples
        self.center_spatial = center_spatial
        
        self._init_variational_parameters()
        
    def _init_variational_parameters(self):
        """Initialize variational distribution parameters."""
        # q(beta) = N(mu_beta, diag(sigma_beta^2))
        self.mu_beta = nn.Parameter(torch.zeros(self.n_features))
        self.log_diag_sigma_beta = nn.Parameter(torch.zeros(self.n_features) - 0.5)
        
        # q(alpha) = N(mu_alpha, diag(sigma_alpha^2)) where phi = U @ alpha
        self.mu_alpha = nn.Parameter(torch.zeros(self.n_obs))
        self.log_sigma_alpha = nn.Parameter(torch.ones(self.n_obs) * (-1.5))
        
        # q(theta) = N(mu_theta, diag(sigma_theta^2))
        init_theta = torch.zeros(self.poly_order + 1)
        init_theta[0] = 0.5
        self.mu_theta = nn.Parameter(init_theta)
        self.log_diag_sigma_theta = nn.Parameter(torch.ones(self.poly_order + 1) * (-2))
        
        # q(tau^2) = InverseGamma(a_tau, b_tau)
        init_a_tau = 5.0
        init_b_tau = (init_a_tau - 1) * 0.5
        self.log_a_tau = nn.Parameter(torch.log(torch.tensor(init_a_tau)))
        self.log_b_tau = nn.Parameter(torch.log(torch.tensor(init_b_tau)))
        
        # q(sigma^2) = InverseGamma(a_sigma, b_sigma)
        init_a_sigma = 6.0
        init_b_sigma = (init_a_sigma - 1) * 0.25
        self.log_a_sigma = nn.Parameter(torch.log(torch.tensor(init_a_sigma)))
        self.log_b_sigma = nn.Parameter(torch.log(torch.tensor(init_b_sigma)))
    
    @property
    def sigma_beta(self) -> torch.Tensor:
        """Standard deviations for beta."""
        return torch.exp(self.log_diag_sigma_beta)
    
    @property
    def sigma_alpha(self) -> torch.Tensor:
        """Standard deviations for alpha (spectral coefficients)."""
        return torch.exp(self.log_sigma_alpha)
    
    @property
    def sigma_theta(self) -> torch.Tensor:
        """Standard deviations for theta."""
        return torch.exp(self.log_diag_sigma_theta)
    
    @property
    def a_tau(self) -> torch.Tensor:
        """Shape parameter for tau^2."""
        return torch.exp(self.log_a_tau)
    
    @property
    def b_tau(self) -> torch.Tensor:
        """Rate parameter for tau^2."""
        return torch.exp(self.log_b_tau)
    
    @property
    def a_sigma(self) -> torch.Tensor:
        """Shape parameter for sigma^2."""
        return torch.exp(self.log_a_sigma)
    
    @property
    def b_sigma(self) -> torch.Tensor:
        """Rate parameter for sigma^2."""
        return torch.exp(self.log_b_sigma)
    
    def sample_variational_params(self, n_samples: int) -> dict:
        """Sample from variational distributions."""
        device = self.mu_beta.device
        
        # Sample beta
        beta_samples = self.mu_beta + self.sigma_beta * torch.randn(
            n_samples, self.n_features, device=device
        )
        
        # Sample alpha (spectral coefficients)
        alpha_samples = self.mu_alpha + self.sigma_alpha * torch.randn(
            n_samples, self.n_obs, device=device
        )
        
        # Apply centering constraint if requested
        if self.center_spatial:
            # Set first component to 0 (corresponds to constant eigenfunction)
            alpha_samples[:, 0] = 0.0
        
        # Transform to spatial domain: phi = U @ alpha
        phi_samples = torch.matmul(alpha_samples, self.eigenvectors.T)
        
        # Sample theta
        theta_samples = self.mu_theta + self.sigma_theta * torch.randn(
            n_samples, self.poly_order + 1, device=device
        )
        
        # Sample tau^2
        gamma_samples = torch.distributions.Gamma(
            self.a_tau, 1.0 / self.b_tau
        ).sample((n_samples,))
        tau2_samples = 1.0 / gamma_samples
        
        # Sample sigma^2
        gamma_samples = torch.distributions.Gamma(
            self.a_sigma, 1.0 / self.b_sigma
        ).sample((n_samples,))
        sigma2_samples = 1.0 / gamma_samples
        
        return {
            'beta': beta_samples,
            'alpha': alpha_samples,
            'phi': phi_samples,
            'theta': theta_samples,
            'tau2': tau2_samples,
            'sigma2': sigma2_samples
        }
    
    def log_likelihood(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        beta: torch.Tensor,
        phi: torch.Tensor,
        sigma2: torch.Tensor
    ) -> torch.Tensor:
        """Compute log p(y | beta, phi, sigma^2)."""
        if beta.dim() == 1:
            # Single sample
            residual = y - X @ beta - phi
            log_lik = -0.5 * self.n_obs * np.log(2 * np.pi)
            log_lik -= 0.5 * self.n_obs * torch.log(sigma2)
            log_lik -= 0.5 * torch.sum(residual**2) / sigma2
        else:
            # Batch of samples
            Xbeta = torch.matmul(beta, X.T)  # (batch, n_obs)
            residual = y.unsqueeze(0) - Xbeta - phi  # (batch, n_obs)
            
            log_lik = -0.5 * self.n_obs * np.log(2 * np.pi)
            log_lik -= 0.5 * self.n_obs * torch.log(sigma2)
            log_lik -= 0.5 * torch.sum(residual**2, dim=1) / sigma2
        
        return log_lik
    
    def log_prior_spatial(
        self,
        alpha: torch.Tensor,
        theta: torch.Tensor,
        tau2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log p(alpha | theta, tau^2) in spectral domain.
        
        In spectral domain: alpha_j ~ N(0, (tau^2 * p(lambda_j; theta))^{-1})
        """
        p_lambda = self.spectral_density(theta)
        p_lambda = torch.clamp(p_lambda, min=1e-6)
        
        if alpha.dim() == 1:
            # Single sample
            precision_alpha = tau2 * p_lambda
            
            if self.center_spatial:
                # Skip first component (constrained to 0)
                log_prior = -0.5 * (self.n_obs - 1) * np.log(2 * np.pi)
                log_prior += 0.5 * torch.sum(torch.log(precision_alpha[1:]))
                log_prior -= 0.5 * torch.sum(precision_alpha[1:] * alpha[1:]**2)
            else:
                log_prior = -0.5 * self.n_obs * np.log(2 * np.pi)
                log_prior += 0.5 * torch.sum(torch.log(precision_alpha))
                log_prior -= 0.5 * torch.sum(precision_alpha * alpha**2)
        else:
            # Batch of samples
            tau2 = tau2.view(-1, 1)
            precision_alpha = tau2 * p_lambda
            
            if self.center_spatial:
                log_prior = -0.5 * (self.n_obs - 1) * np.log(2 * np.pi)
                log_prior += 0.5 * torch.sum(torch.log(precision_alpha[:, 1:]), dim=1)
                log_prior -= 0.5 * torch.sum(precision_alpha[:, 1:] * alpha[:, 1:]**2, dim=1)
            else:
                log_prior = -0.5 * self.n_obs * np.log(2 * np.pi)
                log_prior += 0.5 * torch.sum(torch.log(precision_alpha), dim=1)
                log_prior -= 0.5 * torch.sum(precision_alpha * alpha**2, dim=1)
        
        return log_prior
    
    def entropy_alpha(self) -> torch.Tensor:
        """
        Compute entropy H[q(alpha)] = H[N(mu_alpha, diag(sigma_alpha^2))].
        """
        if self.center_spatial:
            # First component is fixed at 0, contributes 0 entropy
            return 0.5 * (self.n_obs - 1) * (1 + np.log(2 * np.pi)) + torch.sum(self.log_sigma_alpha[1:])
        else:
            return 0.5 * self.n_obs * (1 + np.log(2 * np.pi)) + torch.sum(self.log_sigma_alpha)
    
    def kl_divergence_hyperparameters(self) -> dict:
        """Compute KL divergences for hyperparameters (beta, theta, tau^2, sigma^2)."""
        # KL(q(beta) || p(beta))
        kl_beta = 0.5 * (
            torch.sum(self.sigma_beta**2 / torch.diag(self.prior_beta_cov))
            + torch.sum(((self.mu_beta - self.prior_beta_mean)**2) / torch.diag(self.prior_beta_cov))
            - self.n_features
            + torch.sum(torch.log(torch.diag(self.prior_beta_cov)))
            - 2 * torch.sum(self.log_diag_sigma_beta)
        )
        
        # KL(q(theta) || p(theta))
        kl_theta = 0.5 * (
            torch.sum(self.sigma_theta**2 / torch.diag(self.prior_theta_cov))
            + torch.sum(((self.mu_theta - self.prior_theta_mean)**2) / torch.diag(self.prior_theta_cov))
            - (self.poly_order + 1)
            + torch.sum(torch.log(torch.diag(self.prior_theta_cov)))
            - 2 * torch.sum(self.log_diag_sigma_theta)
        )
        
        # KL(q(tau^2) || p(tau^2))
        kl_tau = (
            self.a_tau * torch.log(self.b_tau / self.prior_tau_b)
            - torch.lgamma(self.a_tau) + torch.lgamma(torch.tensor(self.prior_tau_a))
            + (self.prior_tau_a - self.a_tau) * (torch.log(self.b_tau) - torch.digamma(self.a_tau))
            + self.a_tau * (self.prior_tau_b / self.b_tau - 1.0)
        )
        
        # KL(q(sigma^2) || p(sigma^2))
        kl_sigma = (
            self.a_sigma * torch.log(self.b_sigma / self.prior_sigma_b)
            - torch.lgamma(self.a_sigma) + torch.lgamma(torch.tensor(self.prior_sigma_a))
            + (self.prior_sigma_a - self.a_sigma) * (torch.log(self.b_sigma) - torch.digamma(self.a_sigma))
            + self.a_sigma * (self.prior_sigma_b / self.b_sigma - 1.0)
        )
        
        return {
            'kl_beta': kl_beta,
            'kl_theta': kl_theta,
            'kl_tau': kl_tau,
            'kl_sigma': kl_sigma
        }
    
    def elbo(self, y: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute ELBO with joint inference over hyperparameters and spatial effects.
        
        ELBO = E[log p(y|beta,phi,sigma^2)] + E[log p(phi|theta,tau^2)] 
               + H[q(phi)] - KL[hyperparams]
        """
        # Sample from variational distributions
        samples = self.sample_variational_params(self.n_mc_samples)
        
        # E[log p(y | beta, phi, sigma^2)]
        log_liks = self.log_likelihood(
            y, X,
            samples['beta'],
            samples['phi'],
            samples['sigma2']
        )
        expected_log_lik = torch.mean(log_liks)
        
        # E[log p(phi | theta, tau^2)] in spectral domain
        log_priors_spatial = self.log_prior_spatial(
            samples['alpha'],
            samples['theta'],
            samples['tau2']
        )
        expected_log_prior_spatial = torch.mean(log_priors_spatial)
        
        # H[q(alpha)] - entropy of variational distribution
        entropy_spatial = self.entropy_alpha()
        
        # KL divergences for hyperparameters
        kl_terms = self.kl_divergence_hyperparameters()
        total_kl_hyperparams = sum(kl_terms.values())
        
        # ELBO = E[log p(y|.)] + E[log p(phi|.)] + H[q(phi)] - KL[hyperparams]
        elbo_value = (expected_log_lik + expected_log_prior_spatial + 
                     entropy_spatial - total_kl_hyperparams)
        
        diagnostics = {
            'expected_log_lik': expected_log_lik.item(),
            'expected_log_prior_spatial': expected_log_prior_spatial.item(),
            'entropy_spatial': entropy_spatial.item(),
            'total_kl_hyperparams': total_kl_hyperparams.item(),
            'log_lik_std': torch.std(log_liks).item(),
            **{k: v.item() for k, v in kl_terms.items()}
        }
        
        return elbo_value, diagnostics
    
    def get_spatial_effect_posterior(self, inflate_variance: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get posterior mean and std of spatial effects phi.
        
        Args:
            inflate_variance: Factor to inflate posterior variances (for calibration)
        
        Returns:
            phi_mean: Posterior mean (n_obs,)
            phi_std: Posterior std (n_obs,)
        """
        with torch.no_grad():
            # Mean: phi_mean = U @ mu_alpha
            phi_mean = self.eigenvectors @ self.mu_alpha
            
            # Variance: Var[phi] = U @ diag(sigma_alpha^2) @ U^T
            phi_var = torch.sum(self.eigenvectors**2 * (self.sigma_alpha**2 * inflate_variance), dim=1)
            phi_std = torch.sqrt(phi_var)
            
            return phi_mean, phi_std


class SpectralCARLowRank(SpectralCARBase):
    """
    Spectral CAR model with low-rank covariance structure for spatial effects.
    
    This version adds low-rank structure to the variational posterior:
        q(alpha) ~ N(mu_alpha, Sigma_alpha)
        Sigma_alpha = diag(sigma_alpha^2) + L @ L^T
    
    where L is (n_obs, rank) low-rank factor matrix. This captures posterior
    correlations between spatial components while maintaining O(n*k^2) complexity.
    
    Args:
        (inherits all args from SpectralCARBase)
        low_rank: Rank of low-rank correction (k in Sigma = D + LL^T)
        n_mc_samples: Number of Monte Carlo samples for ELBO
        center_spatial: Whether to enforce mean(phi) = 0
        diagonal_penalty: L2 penalty on diagonal variances (prevents explosion)
        lowrank_penalty: L2 penalty on low-rank factors (prevents dominance)
    """
    
    def __init__(
        self,
        n_obs: int,
        n_features: int,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        poly_order: int = 5,
        low_rank: int = 10,
        n_mc_samples: int = 20,
        center_spatial: bool = True,
        diagonal_penalty: float = 0.001,
        lowrank_penalty: float = 0.0008,
        **kwargs
    ):
        super().__init__(n_obs, n_features, eigenvalues, eigenvectors, poly_order, **kwargs)
        
        self.low_rank = low_rank
        self.n_mc_samples_initial = n_mc_samples
        self.n_mc_samples = n_mc_samples
        self.center_spatial = center_spatial
        self.diagonal_penalty = diagonal_penalty
        self.lowrank_penalty = lowrank_penalty
        
        self._init_variational_parameters()
        
    def _init_variational_parameters(self):
        """Initialize variational parameters with low-rank structure."""
        # q(beta) = N(mu_beta, diag(sigma_beta^2))
        self.mu_beta = nn.Parameter(torch.zeros(self.n_features))
        self.log_diag_sigma_beta = nn.Parameter(torch.zeros(self.n_features) - 0.5)
        
        # q(alpha) = N(mu_alpha, Sigma_alpha) where Sigma_alpha = D + LL^T
        # Mean
        self.mu_alpha = nn.Parameter(torch.zeros(self.n_obs))
        
        # Diagonal component
        self.log_sigma_alpha = nn.Parameter(torch.ones(self.n_obs) * (-1.5))
        
        # Low-rank component: L is (n_obs, rank)
        self.L_alpha = nn.Parameter(torch.randn(self.n_obs, self.low_rank) * 0.01)
        
        # q(theta) = N(mu_theta, diag(sigma_theta^2))
        init_theta = torch.zeros(self.poly_order + 1)
        init_theta[0] = 0.5
        self.mu_theta = nn.Parameter(init_theta)
        self.log_diag_sigma_theta = nn.Parameter(torch.ones(self.poly_order + 1) * (-2))
        
        # q(tau^2) = InverseGamma(a_tau, b_tau)
        init_a_tau = 5.0
        init_b_tau = (init_a_tau - 1) * 0.5
        self.log_a_tau = nn.Parameter(torch.log(torch.tensor(init_a_tau)))
        self.log_b_tau = nn.Parameter(torch.log(torch.tensor(init_b_tau)))
        
        # q(sigma^2) = InverseGamma(a_sigma, b_sigma)
        init_a_sigma = 6.0
        init_b_sigma = (init_a_sigma - 1) * 0.25
        self.log_a_sigma = nn.Parameter(torch.log(torch.tensor(init_a_sigma)))
        self.log_b_sigma = nn.Parameter(torch.log(torch.tensor(init_b_sigma)))
    
    @property
    def sigma_beta(self) -> torch.Tensor:
        return torch.exp(self.log_diag_sigma_beta)
    
    @property
    def sigma_alpha(self) -> torch.Tensor:
        """Diagonal standard deviations."""
        return torch.exp(self.log_sigma_alpha)
    
    @property
    def sigma_theta(self) -> torch.Tensor:
        return torch.exp(self.log_diag_sigma_theta)
    
    @property
    def a_tau(self) -> torch.Tensor:
        return torch.exp(self.log_a_tau)
    
    @property
    def b_tau(self) -> torch.Tensor:
        return torch.exp(self.log_b_tau)
    
    @property
    def a_sigma(self) -> torch.Tensor:
        return torch.exp(self.log_a_sigma)
    
    @property
    def b_sigma(self) -> torch.Tensor:
        return torch.exp(self.log_b_sigma)
    
    def get_alpha_covariance(self) -> torch.Tensor:
        """Compute full covariance matrix Sigma_alpha = D + LL^T."""
        D = torch.diag(self.sigma_alpha**2)
        return D + self.L_alpha @ self.L_alpha.T
    
    def sample_alpha(self, n_samples: int) -> torch.Tensor:
        """
        Sample from q(alpha) with low-rank covariance.
        
        Uses efficient sampling: alpha = mu + D^(1/2) * eps1 + L * eps2
        """
        device = self.mu_alpha.device
        
        # Standard normal samples
        eps1 = torch.randn(n_samples, self.n_obs, device=device)
        eps2 = torch.randn(n_samples, self.low_rank, device=device)
        
        # Transform: alpha = mu + D^(1/2) * eps1 + L * eps2
        alpha_samples = self.mu_alpha.unsqueeze(0)  # (1, n_obs)
        alpha_samples = alpha_samples + self.sigma_alpha * eps1  # Diagonal part
        alpha_samples = alpha_samples + torch.matmul(eps2, self.L_alpha.T)  # Low-rank part
        
        # Apply centering constraint
        if self.center_spatial:
            alpha_samples[:, 0] = 0.0
        
        return alpha_samples
    
    def sample_variational_params(self, n_samples: int) -> dict:
        """Sample from all variational distributions."""
        device = self.mu_beta.device
        
        # Sample beta
        beta_samples = self.mu_beta + self.sigma_beta * torch.randn(
            n_samples, self.n_features, device=device
        )
        
        # Sample alpha with low-rank structure
        alpha_samples = self.sample_alpha(n_samples)
        
        # Transform to spatial domain: phi = U @ alpha
        phi_samples = torch.matmul(alpha_samples, self.eigenvectors.T)
        
        # Sample theta
        theta_samples = self.mu_theta + self.sigma_theta * torch.randn(
            n_samples, self.poly_order + 1, device=device
        )
        
        # Sample tau^2
        gamma_samples = torch.distributions.Gamma(
            self.a_tau, 1.0 / self.b_tau
        ).sample((n_samples,))
        tau2_samples = 1.0 / gamma_samples
        
        # Sample sigma^2
        gamma_samples = torch.distributions.Gamma(
            self.a_sigma, 1.0 / self.b_sigma
        ).sample((n_samples,))
        sigma2_samples = 1.0 / gamma_samples
        
        return {
            'beta': beta_samples,
            'alpha': alpha_samples,
            'phi': phi_samples,
            'theta': theta_samples,
            'tau2': tau2_samples,
            'sigma2': sigma2_samples
        }
    
    def log_likelihood(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        beta: torch.Tensor,
        phi: torch.Tensor,
        sigma2: torch.Tensor
    ) -> torch.Tensor:
        """Compute log p(y | beta, phi, sigma^2)."""
        if beta.dim() == 1:
            residual = y - X @ beta - phi
            log_lik = -0.5 * self.n_obs * np.log(2 * np.pi)
            log_lik -= 0.5 * self.n_obs * torch.log(sigma2)
            log_lik -= 0.5 * torch.sum(residual**2) / sigma2
        else:
            Xbeta = torch.matmul(beta, X.T)
            residual = y.unsqueeze(0) - Xbeta - phi
            
            log_lik = -0.5 * self.n_obs * np.log(2 * np.pi)
            log_lik -= 0.5 * self.n_obs * torch.log(sigma2)
            log_lik -= 0.5 * torch.sum(residual**2, dim=1) / sigma2
        
        return log_lik
    
    def log_prior_spatial(
        self,
        alpha: torch.Tensor,
        theta: torch.Tensor,
        tau2: torch.Tensor
    ) -> torch.Tensor:
        """Compute log p(alpha | theta, tau^2) in spectral domain."""
        p_lambda = self.spectral_density(theta)
        p_lambda = torch.clamp(p_lambda, min=1e-6)
        
        if alpha.dim() == 1:
            precision_alpha = tau2 * p_lambda
            
            if self.center_spatial:
                log_prior = -0.5 * (self.n_obs - 1) * np.log(2 * np.pi)
                log_prior += 0.5 * torch.sum(torch.log(precision_alpha[1:]))
                log_prior -= 0.5 * torch.sum(precision_alpha[1:] * alpha[1:]**2)
            else:
                log_prior = -0.5 * self.n_obs * np.log(2 * np.pi)
                log_prior += 0.5 * torch.sum(torch.log(precision_alpha))
                log_prior -= 0.5 * torch.sum(precision_alpha * alpha**2)
        else:
            tau2 = tau2.view(-1, 1)
            precision_alpha = tau2 * p_lambda
            
            if self.center_spatial:
                log_prior = -0.5 * (self.n_obs - 1) * np.log(2 * np.pi)
                log_prior += 0.5 * torch.sum(torch.log(precision_alpha[:, 1:]), dim=1)
                log_prior -= 0.5 * torch.sum(precision_alpha[:, 1:] * alpha[:, 1:]**2, dim=1)
            else:
                log_prior = -0.5 * self.n_obs * np.log(2 * np.pi)
                log_prior += 0.5 * torch.sum(torch.log(precision_alpha), dim=1)
                log_prior -= 0.5 * torch.sum(precision_alpha * alpha**2, dim=1)
        
        return log_prior
    
    def entropy_alpha(self) -> torch.Tensor:
        """
        Compute entropy H[q(alpha)] with low-rank structure.
        
        Uses matrix determinant lemma:
        log |D + LL^T| = log |D| + log |I + L^T D^(-1) L|
        """
        # Diagonal part
        log_det_D = 2 * torch.sum(self.log_sigma_alpha)
        
        # Low-rank correction
        D_inv_sqrt = 1.0 / (self.sigma_alpha + 1e-8)
        D_inv_sqrt_L = D_inv_sqrt.unsqueeze(-1) * self.L_alpha  # (n, k)
        
        M = torch.eye(self.low_rank, device=self.L_alpha.device) + torch.matmul(D_inv_sqrt_L.T, D_inv_sqrt_L)
        M = M + torch.eye(self.low_rank, device=M.device) * 1e-6  # Numerical stability
        log_det_correction = torch.logdet(M)
        
        log_det_Sigma = log_det_D + log_det_correction
        
        # Account for centering
        if self.center_spatial:
            n_free = self.n_obs - 1
        else:
            n_free = self.n_obs
        
        # Entropy with dual regularization
        entropy = 0.5 * n_free * (1 + np.log(2 * np.pi)) + 0.5 * log_det_Sigma
        
        # Regularization penalties
        diagonal_reg = self.diagonal_penalty * torch.sum(self.sigma_alpha**2)
        lowrank_reg = self.lowrank_penalty * torch.sum(self.L_alpha**2)
        
        entropy_regularized = entropy - diagonal_reg - lowrank_reg
        
        return entropy_regularized
    
    def kl_divergence_hyperparameters(self) -> dict:
        """Compute KL divergences for hyperparameters."""
        # KL(q(beta) || p(beta))
        kl_beta = 0.5 * (
            torch.sum(self.sigma_beta**2 / torch.diag(self.prior_beta_cov))
            + torch.sum(((self.mu_beta - self.prior_beta_mean)**2) / torch.diag(self.prior_beta_cov))
            - self.n_features
            + torch.sum(torch.log(torch.diag(self.prior_beta_cov)))
            - 2 * torch.sum(self.log_diag_sigma_beta)
        )
        
        # KL(q(theta) || p(theta))
        kl_theta = 0.5 * (
            torch.sum(self.sigma_theta**2 / torch.diag(self.prior_theta_cov))
            + torch.sum(((self.mu_theta - self.prior_theta_mean)**2) / torch.diag(self.prior_theta_cov))
            - (self.poly_order + 1)
            + torch.sum(torch.log(torch.diag(self.prior_theta_cov)))
            - 2 * torch.sum(self.log_diag_sigma_theta)
        )
        
        # KL(q(tau^2) || p(tau^2))
        kl_tau = (
            self.a_tau * torch.log(self.b_tau / self.prior_tau_b)
            - torch.lgamma(self.a_tau) + torch.lgamma(torch.tensor(self.prior_tau_a))
            + (self.prior_tau_a - self.a_tau) * (torch.log(self.b_tau) - torch.digamma(self.a_tau))
            + self.a_tau * (self.prior_tau_b / self.b_tau - 1.0)
        )
        
        # KL(q(sigma^2) || p(sigma^2))
        kl_sigma = (
            self.a_sigma * torch.log(self.b_sigma / self.prior_sigma_b)
            - torch.lgamma(self.a_sigma) + torch.lgamma(torch.tensor(self.prior_sigma_a))
            + (self.prior_sigma_a - self.a_sigma) * (torch.log(self.b_sigma) - torch.digamma(self.a_sigma))
            + self.a_sigma * (self.prior_sigma_b / self.b_sigma - 1.0)
        )
        
        return {
            'kl_beta': kl_beta,
            'kl_theta': kl_theta,
            'kl_tau': kl_tau,
            'kl_sigma': kl_sigma
        }
    
    def elbo(self, y: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute ELBO with low-rank covariance structure."""
        # Sample from variational distributions
        samples = self.sample_variational_params(self.n_mc_samples)
        
        # E[log p(y | beta, phi, sigma^2)]
        log_liks = self.log_likelihood(
            y, X,
            samples['beta'],
            samples['phi'],
            samples['sigma2']
        )
        expected_log_lik = torch.mean(log_liks)
        
        # E[log p(phi | theta, tau^2)] in spectral domain
        log_priors_spatial = self.log_prior_spatial(
            samples['alpha'],
            samples['theta'],
            samples['tau2']
        )
        expected_log_prior_spatial = torch.mean(log_priors_spatial)
        
        # H[q(alpha)] with low-rank structure (includes regularization)
        entropy_spatial = self.entropy_alpha()
        
        # KL divergences for hyperparameters
        kl_terms = self.kl_divergence_hyperparameters()
        total_kl_hyperparams = sum(kl_terms.values())
        
        # ELBO
        elbo_value = (expected_log_lik + expected_log_prior_spatial + 
                     entropy_spatial - total_kl_hyperparams)
        
        diagnostics = {
            'expected_log_lik': expected_log_lik.item(),
            'expected_log_prior_spatial': expected_log_prior_spatial.item(),
            'entropy_spatial': entropy_spatial.item(),
            'total_kl_hyperparams': total_kl_hyperparams.item(),
            'log_lik_std': torch.std(log_liks).item(),
            'diagonal_penalty_value': (self.diagonal_penalty * torch.sum(self.sigma_alpha**2)).item(),
            'lowrank_penalty_value': (self.lowrank_penalty * torch.sum(self.L_alpha**2)).item(),
            **{k: v.item() for k, v in kl_terms.items()}
        }
        
        return elbo_value, diagnostics
    
    def get_spatial_effect_posterior(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get posterior mean and std of spatial effects with low-rank covariance.
        """
        with torch.no_grad():
            # Mean
            phi_mean = self.eigenvectors @ self.mu_alpha
            
            # Variance: Var[phi] = U @ Sigma_alpha @ U^T
            # Diagonal part: U @ D @ U^T
            var_diagonal = torch.sum(self.eigenvectors**2 * self.sigma_alpha**2, dim=1)
            
            # Low-rank part: U @ (LL^T) @ U^T = (U @ L) @ (U @ L)^T
            UL = self.eigenvectors @ self.L_alpha  # (n, k)
            var_lowrank = torch.sum(UL**2, dim=1)
            
            phi_var = var_diagonal + var_lowrank
            phi_std = torch.sqrt(phi_var)
            
            return phi_mean, phi_std
    
    def get_parameter_summary(self) -> dict:
        """Get summary of estimated parameters including low-rank info."""
        summary = super().get_parameter_summary()
        
        with torch.no_grad():
            summary['lowrank_contribution'] = torch.norm(self.L_alpha).item()
            
        return summary