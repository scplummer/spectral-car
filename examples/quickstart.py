"""
Quick start example for Spectral CAR models.

Minimal example showing the essential workflow in ~30 lines.
"""

import torch
from spectral_car.utils import create_grid_graph_laplacian, generate_benchmark_dataset
from spectral_car.models import SpectralCARMeanField
from spectral_car.inference import VariationalInference

# Set random seed
torch.manual_seed(42)

# 1. Create spatial graph and generate data
print("Generating data...")
grid_size = 10
n_obs = grid_size ** 2
eigenvalues, eigenvectors = create_grid_graph_laplacian(n_obs, grid_size)
data = generate_benchmark_dataset('smooth', n_obs, grid_size, eigenvalues, eigenvectors)
y, X = data['y'], data['X']

# 2. Initialize model
print("Initializing model...")
model = SpectralCARMeanField(
    n_obs=n_obs,
    n_features=X.shape[1],
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    poly_order=5
)

# 3. Fit with variational inference
print("Fitting model...")
vi = VariationalInference(model)
history = vi.fit(y, X, n_iterations=500, learning_rate=0.01, verbose=True, print_every=100)

# 4. Make predictions
print("\nMaking predictions...")
with torch.no_grad():
    beta_mean = model.mu_beta
    
    if hasattr(model, 'mu_alpha'):
        # Has explicit spatial coefficients
        phi_estimate = model.eigenvectors @ model.mu_alpha
    else:
        # Marginalized model - compute posterior mean from residuals
        residual = y - X @ beta_mean
        residual_spectral = model.eigenvectors.T @ residual
        
        # Get spectral density p(λ)
        theta = model.mu_theta
        T = torch.zeros(n_obs, model.poly_order + 1)
        T[:, 0] = 1.0
        if model.poly_order >= 1:
            T[:, 1] = model.eigenvalues_normalized
        for k in range(2, model.poly_order + 1):
            T[:, k] = 2 * model.eigenvalues_normalized * T[:, k-1] - T[:, k-2]
        spectral_density = torch.exp(T @ theta)
        spectral_density = torch.clamp(spectral_density, min=1e-6)
        
        # Get variance parameters
        tau2 = (model.b_tau / (model.a_tau - 1))
        sigma2 = (model.b_sigma / (model.a_sigma - 1))
        
        # Posterior mean in spectral domain
        # E[α_j | y] = (τ²p(λ_j)) / (σ² + τ²p(λ_j)) * ỹ_j
        posterior_var = tau2 * spectral_density
        posterior_weight = posterior_var / (sigma2 + posterior_var)
        alpha_mean = posterior_weight * residual_spectral
        
        # Transform back to spatial domain
        phi_estimate = model.eigenvectors @ alpha_mean
    
    predictions = X @ beta_mean + phi_estimate
    
    predictions = X @ beta_mean + phi_estimate

# 5. Evaluate
rmse = ((predictions - y) ** 2).mean().sqrt()

# Debug: Check if phi values are valid
print(f"phi_estimate stats: min={phi_estimate.min():.3f}, max={phi_estimate.max():.3f}, mean={phi_estimate.mean():.3f}")
print(f"phi_true stats: min={data['phi'].min():.3f}, max={data['phi'].max():.3f}, mean={data['phi'].mean():.3f}")

# Check for NaN or constant values
if torch.isnan(phi_estimate).any() or torch.isnan(data['phi']).any():
    phi_corr = float('nan')
    print("Warning: NaN values detected in spatial fields")
elif phi_estimate.std() < 1e-6 or data['phi'].std() < 1e-6:
    phi_corr = float('nan')
    print("Warning: Spatial field has near-zero variance")
else:
    phi_corr = torch.corrcoef(torch.stack([phi_estimate, data['phi']]))[0, 1]

print(f"\n{'='*40}")
print(f"Results:")
print(f"  RMSE: {rmse:.4f}")
print(f"  Spatial field correlation: {phi_corr:.4f}")
print(f"  Final ELBO: {history[-1]['elbo']:.2f}")
print(f"{'='*40}")