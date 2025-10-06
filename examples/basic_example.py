"""
Basic usage example for Spectral CAR models.

This script demonstrates:
1. Creating synthetic spatial data
2. Fitting a Spectral CAR model with variational inference
3. Making predictions
4. Visualizing results
5. Model diagnostics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from spectral_car.utils import (
    create_grid_graph_laplacian,
    generate_benchmark_dataset,
    compute_coverage,
    compute_prediction_intervals
)
from spectral_car.models import SpectralCARMeanField
from spectral_car.inference import VariationalInference
from spectral_car.visualization import (
    plot_spatial_field,
    plot_convergence,
    plot_spectral_density
)


def main():
    """Run basic example workflow."""
    
    print("=" * 60)
    print("Spectral CAR Model - Basic Example")
    print("=" * 60)
    
    # ========================================================================
    # 1. Generate Synthetic Data
    # ========================================================================
    print("\n1. Generating synthetic spatial data...")
    
    # Create a 10x10 grid (100 locations)
    grid_size = 10
    n_obs = grid_size ** 2
    
    # Get graph Laplacian eigendecomposition
    eigenvalues, eigenvectors = create_grid_graph_laplacian(n_obs, grid_size)
    print(f"   Grid: {grid_size}x{grid_size} ({n_obs} locations)")
    print(f"   Eigenvalues range: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
    
    # Generate benchmark dataset with smooth spatial field
    data = generate_benchmark_dataset(
        'smooth',  # Try 'smooth', 'rough', 'multi_scale', or 'anisotropic'
        n_obs=n_obs,
        grid_size=grid_size,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        seed=42
    )
    
    y = data['y']
    X = data['X']
    phi_true = data['phi']
    
    print(f"   Observations: {y.shape}")
    print(f"   Covariates: {X.shape}")
    print(f"   True spatial field range: [{phi_true.min():.3f}, {phi_true.max():.3f}]")
    
    # ========================================================================
    # 2. Initialize and Fit Model
    # ========================================================================
    print("\n2. Fitting Spectral CAR model...")
    
    # Initialize model with parametric spectral form
    model = SpectralCARMeanField(
        n_obs=n_obs,
        n_features=X.shape[1],
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        #poly_order=5,  # Only used if spectral_form='chebyshev'
        spectral_form='rational',  # Choose: 'rational', 'exponential', 'power_law', or 'chebyshev' 
        prior_beta_std=10.0,
        prior_theta_std=0.5,
        prior_tau_a=3.0,
        prior_tau_b=1.0,
        prior_sigma_a=5.0,
        prior_sigma_b=1.0
    )
    
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Spectral form: {model.spectral_form}")
    print(f"   Theta dimension: {len(model.mu_theta)}")
    print(f"   Eigenvalue range: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
    print(f"   Normalized range: [{model.eigenvalues_normalized.min():.3f}, {model.eigenvalues_normalized.max():.3f}]")
    
    # Initialize inference engine
    vi = VariationalInference(model=model)
    
    # Fit model
    print("   Training...")
    history = vi.fit(
        y=y,
        X=X,
        n_iterations=5000,
        learning_rate=1e-3,
        verbose=True,
        print_every=1000
    )
    
    print(f"\n   Training complete!")
    print(f"   Final ELBO: {history[-1]['elbo']:.2f}")
    print(f"   Iterations: {len(history)}")
    
    # ========================================================================
    # 3. Make Predictions
    # ========================================================================
    print("\n3. Making predictions...")
    
    # Posterior mean predictions
    with torch.no_grad():
        # Get posterior means
        beta_mean = model.mu_beta
        
        # Check if model has explicit spatial effects
        if hasattr(model, 'mu_alpha'):
            # Model has spectral coefficients - transform to spatial domain
            phi_mean = model.eigenvectors @ model.mu_alpha
        else:
            # MeanField model - compute posterior mean from residuals
            residual = y - X @ beta_mean
            residual_spectral = model.eigenvectors.T @ residual
            
            # Get spectral density p(λ) using model's method
            theta = model.mu_theta
            spectral_density = model.spectral_density(theta)
            
            # Get variance parameters
            tau2 = (model.b_tau / (model.a_tau - 1))
            sigma2 = (model.b_sigma / (model.a_sigma - 1))
            
            # Posterior mean in spectral domain
            posterior_var = tau2 * spectral_density
            posterior_weight = posterior_var / (sigma2 + posterior_var)
            alpha_mean = posterior_weight * residual_spectral
            
            # Transform back to spatial domain
            phi_mean = model.eigenvectors @ alpha_mean
        
        # Predictions: y_hat = X @ beta + phi
        predictions = X @ beta_mean + phi_mean
        phi_est = phi_mean
    
    print(f"   Predictions: {predictions.shape}")
    print(f"   Estimated spatial field: {phi_est.shape}")
    
    # Sample from posterior for uncertainty quantification
    n_samples = 1000
    
    # Sample spatial field
    if hasattr(model, 'mu_alpha') and hasattr(model, 'sigma_alpha'):
        # Sample in spectral domain then transform to spatial
        alpha_samples = model.mu_alpha + model.sigma_alpha * torch.randn(n_samples, n_obs)
        phi_samples = (model.eigenvectors @ alpha_samples.T).T  # (n_samples, n_obs)
    else:
        # For marginalized models, sample using posterior variance
        theta = model.mu_theta
        spectral_density = model.spectral_density(theta)
        
        tau2 = (model.b_tau / (model.a_tau - 1))
        sigma2 = (model.b_sigma / (model.a_sigma - 1))
        
        # Posterior variance in spectral domain
        posterior_var = (tau2 * spectral_density * sigma2) / (sigma2 + tau2 * spectral_density)
        posterior_std = torch.sqrt(posterior_var)
        
        # Sample alpha in spectral domain
        residual = y - X @ beta_mean
        residual_spectral = model.eigenvectors.T @ residual
        posterior_weight = (tau2 * spectral_density) / (sigma2 + tau2 * spectral_density)
        alpha_mean = posterior_weight * residual_spectral
        
        alpha_samples = alpha_mean + posterior_std * torch.randn(n_samples, n_obs)
        phi_samples = (model.eigenvectors @ alpha_samples.T).T  # (n_samples, n_obs)
    
    # Compute prediction intervals
    lower, upper = compute_prediction_intervals(
        phi_samples, 
        coverage=0.95, 
        method='quantile'
    )
    
    print(f"   Generated {n_samples} posterior samples")
    
    # ========================================================================
    # 4. Evaluate Performance
    # ========================================================================
    print("\n4. Model evaluation...")
    
    # Prediction error
    mae = (predictions - y).abs().mean()
    rmse = ((predictions - y) ** 2).mean().sqrt()
    
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    
    # Spatial field recovery
    phi_mae = (phi_est - phi_true).abs().mean()
    phi_correlation = torch.corrcoef(torch.stack([phi_est, phi_true]))[0, 1]
    
    print(f"   Spatial field MAE: {phi_mae:.4f}")
    print(f"   Spatial field correlation: {phi_correlation:.4f}")
    
    # Coverage statistics
    coverage_stats = compute_coverage(
        predictions=phi_est,
        lower=lower,
        upper=upper,
        true_values=phi_true
    )
    
    print(f"   95% interval coverage: {coverage_stats['coverage']:.3f}")
    print(f"   Mean interval width: {coverage_stats['mean_interval_width']:.3f}")
    
    # Extract parameter estimates
    beta_mean = model.mu_beta.detach()
    theta_mean = model.mu_theta.detach()
    tau2_mean = (model.b_tau / (model.a_tau - 1)).detach()
    sigma2_mean = (model.b_sigma / (model.a_sigma - 1)).detach()
    
    print(f"\n   Estimated parameters:")
    print(f"   β (coefficients): {beta_mean.numpy()}")
    print(f"   θ (spectral): {theta_mean.numpy()}")
    print(f"   τ² (spatial variance): {tau2_mean:.4f}")
    print(f"   σ² (noise variance): {sigma2_mean:.4f}")
    
    # ========================================================================
    # 5. Visualize Results
    # ========================================================================
    print("\n5. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot true spatial field
    ax = axes[0, 0]
    phi_true_grid = phi_true.detach().numpy().reshape(grid_size, grid_size)
    im = ax.imshow(phi_true_grid, cmap='RdBu_r', origin='lower')
    ax.set_title('True Spatial Field')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    
    # Plot estimated spatial field
    ax = axes[0, 1]
    phi_est_grid = phi_est.detach().numpy().reshape(grid_size, grid_size)
    im = ax.imshow(phi_est_grid, cmap='RdBu_r', origin='lower')
    ax.set_title('Estimated Spatial Field')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    
    # Plot estimation error
    ax = axes[0, 2]
    error_grid = (phi_est - phi_true).detach().numpy().reshape(grid_size, grid_size)
    im = ax.imshow(error_grid, cmap='RdBu_r', origin='lower')
    ax.set_title('Estimation Error')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    
    # Plot ELBO convergence
    ax = axes[1, 0]
    elbos = [h['elbo'] for h in history]
    ax.plot(elbos, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ELBO')
    ax.set_title('Training Convergence')
    ax.grid(True, alpha=0.3)
    
    # Plot uncertainty (interval width)
    ax = axes[1, 1]
    width_grid = (upper - lower).detach().reshape(grid_size, grid_size)
    im = ax.imshow(width_grid.numpy(), cmap='viridis', origin='lower')
    ax.set_title('95% Interval Width')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    
    # Plot spectral density - true vs estimated
    ax = axes[1, 2]
    
    # Compute estimated spectral density using model's method
    with torch.no_grad():
        theta_est = model.mu_theta
        tau2_est = (model.b_tau / (model.a_tau - 1))
        spectral_density_est = model.spectral_density(theta_est)
        precision_est = tau2_est * spectral_density_est
        
        # Compute true spectral density if available AND forms match
        if (data['theta'] is not None and 
            data.get('spectral_form') == model.spectral_form):
            theta_true = data['theta']
            tau2_true = data['tau2']
            
            # Check dimensions match
            if len(theta_true) != len(theta_est):
                print(f"\nWarning: theta dimension mismatch - true: {len(theta_true)}, est: {len(theta_est)}")
                # Pad or truncate theta_true to match
                if len(theta_true) < len(theta_est):
                    theta_true = torch.cat([theta_true, torch.zeros(len(theta_est) - len(theta_true))])
                else:
                    theta_true = theta_true[:len(theta_est)]
            
            spectral_density_true = model.spectral_density(theta_true)
            precision_true = tau2_true * spectral_density_true
            
            # Print for debugging
            print(f"\nθ comparison:")
            print(f"  True:      {theta_true.numpy()}")
            print(f"  Estimated: {theta_est.numpy()}")
            
            # Plot both
            ax.plot(eigenvalues.detach().numpy(), precision_true.detach().numpy(), 
                'k--', linewidth=2, label='True precision', alpha=0.7)
            ax.plot(eigenvalues.detach().numpy(), precision_est.detach().numpy(), 
                'b-', linewidth=2, label='Estimated precision')
            ax.legend(loc='best')
        else:
            # Just plot estimated
            ax.plot(eigenvalues.detach().numpy(), precision_est.detach().numpy(), 
                'b-', linewidth=2, label='Estimated precision')
            if data['theta'] is not None:
                ax.text(0.5, 0.95, f"Data form: {data.get('spectral_form', 'unknown')}\nModel form: {model.spectral_form}",
                        transform=ax.transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Precision (τ² × p(λ))')
    ax.set_title(f'Spectral Density ({model.spectral_form})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('spectral_car_example.png', dpi=150, bbox_inches='tight')
    print("   Saved figure: spectral_car_example.png")
    
    # ========================================================================
    # 6. Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f" Model fitted successfully")
    print(f" Spectral form: {model.spectral_form}")
    print(f" RMSE: {rmse:.4f}")
    print(f" Spatial field correlation: {phi_correlation:.4f}")
    print(f" Coverage: {coverage_stats['coverage']:.3f}")
    print(f" Visualizations saved")
    print("=" * 60)
    
    return model, data, history


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run example
    model, data, history = main()