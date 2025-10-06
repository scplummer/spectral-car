# Spectral CAR: Spectral Conditional Autoregressive Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for Bayesian spatial modeling using spectral representations of Conditional Autoregressive (CAR) models with variational inference.

## Overview

**Spectral CAR** provides efficient Bayesian inference for spatial data by:
- Representing spatial dependencies through graph Laplacian eigendecomposition
- Using Chebyshev polynomial approximations for flexible spectral densities
- Implementing scalable variational inference with optional calibration
- Supporting multiple model variants (mean-field, low-rank, joint)

### Key Features

- **Spectral representation**: Transform spatial models to the frequency domain for efficient computation
- **Flexible inference**: Mean-field and low-rank variational approximations
- **Uncertainty quantification**: Post-hoc calibration for accurate prediction intervals
- **Scalable**: Handles moderate to large spatial datasets (100-10,000+ locations)
- **Well-tested**: Comprehensive synthetic data generation and diagnostic tools

---

## Installation

### From source (development mode)

```bash
# Clone the repository
git clone https://github.com/yourusername/spectral-car.git
cd spectral-car

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- SciPy
- Matplotlib

All dependencies are automatically installed via `pip install -e .`

---

## Quick Start

```python
import torch
from spectral_car import SpectralCARMeanField, VariationalInference
from spectral_car.utils import create_grid_graph_laplacian, generate_benchmark_dataset

# 1. Create spatial graph (10x10 grid)
eigenvalues, eigenvectors = create_grid_graph_laplacian(100, 10)

# 2. Generate synthetic data
data = generate_benchmark_dataset('smooth', 100, 10, eigenvalues, eigenvectors)
y, X = data['y'], data['X']

# 3. Initialize and fit model
model = SpectralCARMeanField(
    n_obs=100, 
    n_features=3,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors
)

vi = VariationalInference(model)
history = vi.fit(y, X, n_iterations=500, learning_rate=0.01)

# 4. Make predictions
with torch.no_grad():
    predictions = X @ model.mu_beta
    print(f"RMSE: {((predictions - y) ** 2).mean().sqrt():.3f}")
```

See `examples/` for complete working examples.

---

## Model Variants

### SpectralCARMeanField

Marginalizes out spatial effects for computational efficiency. Best for:
- Large datasets (1000+ locations)
- Fast inference needed
- Don't need explicit spatial field estimates

```python
model = SpectralCARMeanField(
    n_obs=n_obs,
    n_features=n_features,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    poly_order=5
)
```

### SpectralCARLowRank

Low-rank approximation for spatial effects. Best for:
- Medium datasets (500-5000 locations)
- Balance between speed and accuracy
- Want spatial field estimates with reduced computation

```python
model = SpectralCARLowRank(
    n_obs=n_obs,
    n_features=n_features,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    rank=20,  # Low-rank dimension
    poly_order=5
)
```

### SpectralCARCollapsed (Joint)

Explicit spatial effects in variational family. Best for:
- Smaller datasets (<1000 locations)
- Need detailed spatial field uncertainty
- Highest accuracy needed

```python
model = SpectralCARCollapsed(
    n_obs=n_obs,
    n_features=n_features,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    poly_order=5
)
```

---

## Inference

### Standard Variational Inference

```python
from spectral_car import VariationalInference

vi = VariationalInference(model)
history = vi.fit(
    y, X,
    n_iterations=1000,
    learning_rate=0.01,
    n_mc_samples_final=50,  # Ramp up MC samples during training
    warmup_iterations=200,
    use_scheduler=True,
    verbose=True
)
```

**Key parameters:**
- `n_iterations`: Number of optimization steps
- `learning_rate`: Initial learning rate (Adam optimizer)
- `n_mc_samples_final`: Target number of Monte Carlo samples for ELBO (gradually increased)
- `warmup_iterations`: Iterations before ramping MC samples
- `use_scheduler`: Use ReduceLROnPlateau scheduler

### Calibrated Variational Inference

Improves uncertainty quantification using validation data:

```python
from spectral_car import CalibratedVI

vi = CalibratedVI(model, target_coverage=0.95)

# Fit with automatic calibration
history = vi.fit(
    y_train, X_train,
    y_val, X_val,  # Validation data for calibration
    auto_calibrate=True,
    n_iterations=1000,
    learning_rate=0.01
)

# Get calibrated predictions
predictions, uncertainties = vi.predict(y_test, X_test, calibrated=True)
lower, upper = predictions - 1.96 * uncertainties, predictions + 1.96 * uncertainties
```

---

## Utilities

### Graph Construction

Create spatial graphs from coordinates or grid structures:

```python
from spectral_car.utils import (
    create_grid_graph_laplacian,
    create_adjacency_from_coords,
    create_laplacian_from_adjacency
)

# Regular grid
eigenvalues, eigenvectors = create_grid_graph_laplacian(n_nodes=100, grid_size=10)

# From coordinates (k-nearest neighbors)
coords = torch.randn(100, 2)  # 100 locations in 2D
adjacency = create_adjacency_from_coords(coords, adjacency_type='knn', k=5)
eigenvalues, eigenvectors = create_laplacian_from_adjacency(adjacency)

# From coordinates (distance threshold)
adjacency = create_adjacency_from_coords(coords, adjacency_type='threshold', threshold=0.5)
```

### Data Generation

Generate synthetic spatial data for testing:

```python
from spectral_car.utils import generate_benchmark_dataset

# Pre-configured benchmark datasets
data = generate_benchmark_dataset(
    'smooth',  # or 'rough', 'multi_scale', 'anisotropic'
    n_obs=100,
    grid_size=10,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    seed=42
)

y = data['y']           # Observations
X = data['X']           # Covariates
phi_true = data['phi']  # True spatial field
```

### Model Diagnostics

```python
from spectral_car.utils import (
    compute_waic,
    compute_loo,
    compare_models,
    compute_prediction_intervals,
    compute_coverage
)

# Information criteria
log_liks = model.compute_log_likelihood(y, X, n_samples=1000)
waic = compute_waic(log_liks)
loo = compute_loo(log_liks)

# Model comparison
models = {
    'meanfield': model1,
    'lowrank': model2,
    'joint': model3
}
comparison = compare_models(models, y, X, criterion='waic')
print(comparison['ranking'])

# Coverage diagnostics
phi_samples = model.sample_spatial_field(1000)
lower, upper = compute_prediction_intervals(phi_samples, coverage=0.95)
stats = compute_coverage(predictions, lower, upper, y_true)
print(f"Coverage: {stats['coverage']:.1%}")
```

---

## Examples

### Minimal Example (30 lines)

See `examples/quickstart.py` for a minimal working example.

### Full Example with Visualization

See `examples/basic_example.py` for a comprehensive demo that includes:
- Data generation with multiple benchmark datasets
- Model fitting with progress tracking
- Predictions and uncertainty quantification
- Performance evaluation (RMSE, MAE, coverage)
- 6-panel diagnostic visualizations
- Parameter extraction and interpretation

Run with:
```bash
cd examples
python basic_example.py
```

Output:
- Console output with training progress and diagnostics
- Saved figure: `spectral_car_example.png` with spatial field maps, convergence plots, and uncertainty quantification

---

## Mathematical Model

### Basic CAR Model

The full hierarchical model is:

$$
\begin{aligned}
y_i &= \mathbf{x}_i^\top \boldsymbol{\beta} + \phi_i + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma^2) \\
\boldsymbol{\phi} &\sim N(\mathbf{0}, \tau^2 (D - \rho W)^{-1}) \\
\boldsymbol{\beta} &\sim N(\mathbf{0}, \sigma_\beta^2 I) \\
\tau^2 &\sim \text{InverseGamma}(a_\tau, b_\tau) \\
\sigma^2 &\sim \text{InverseGamma}(a_\sigma, b_\sigma)
\end{aligned}
$$

Where:
- $y_i$ = observation at location $i$
- $\mathbf{x}_i$ = covariate vector at location $i$
- $\boldsymbol{\beta}$ = fixed effect coefficients
- $\phi_i$ = spatial random effect at location $i$
- $W$ = spatial adjacency matrix
- $D$ = degree matrix (diagonal with row sums of $W$)
- $\tau^2$ = spatial variance parameter
- $\sigma^2$ = observation noise variance

### Spectral Representation

Transform to the spectral domain using the graph Laplacian eigendecomposition:

$$
L = D - W = U \Lambda U^\top
$$

Where $U$ is the matrix of eigenvectors and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ contains the eigenvalues.

The spatial field in the spectral domain:

$$
\boldsymbol{\alpha} = U^\top \boldsymbol{\phi}, \quad \alpha_j \sim N(0, \tau^2 / \lambda_j)
$$

Each spectral coefficient $\alpha_j$ is independent with variance inversely proportional to the eigenvalue $\lambda_j$.

### Flexible Spectral Density

Rather than fixed precision $\lambda_j / \tau^2$, we use a Chebyshev polynomial approximation:

$$
p(\lambda_j; \boldsymbol{\theta}) = \exp\left(\sum_{k=0}^K \theta_k T_k(\tilde{\lambda}_j)\right)
$$

Where:
- $T_k(x)$ = Chebyshev polynomial of order $k$
- $\tilde{\lambda}_j = 2(\lambda_j - \lambda_{\min}) / (\lambda_{\max} - \lambda_{\min}) - 1$ (normalized to $[-1, 1]$)
- $\boldsymbol{\theta} = (\theta_0, \ldots, \theta_K)$ = spectral coefficients to be learned

The spatial prior becomes:

$$
\alpha_j \sim N(0, \tau^2 / p(\lambda_j; \boldsymbol{\theta}))
$$

This allows the model to learn arbitrary smoothness patterns across spatial frequencies.

### Variational Approximation

We approximate the posterior with a mean-field factorization:

$$
q(\boldsymbol{\beta}, \boldsymbol{\theta}, \tau^2, \sigma^2) = q(\boldsymbol{\beta}) q(\boldsymbol{\theta}) q(\tau^2) q(\sigma^2)
$$

With:
- $q(\boldsymbol{\beta}) = N(\boldsymbol{\mu}_\beta, \text{diag}(\boldsymbol{\sigma}_\beta^2))$
- $q(\boldsymbol{\theta}) = N(\boldsymbol{\mu}_\theta, \text{diag}(\boldsymbol{\sigma}_\theta^2))$
- $q(\tau^2) = \text{InverseGamma}(a_\tau^q, b_\tau^q)$
- $q(\sigma^2) = \text{InverseGamma}(a_\sigma^q, b_\sigma^q)$

The Evidence Lower Bound (ELBO):

$$
\mathcal{L} = \mathbb{E}_q[\log p(\mathbf{y} | \boldsymbol{\beta}, \boldsymbol{\theta}, \tau^2, \sigma^2)] - \text{KL}(q \| p)
$$

Optimized using stochastic gradient ascent with Monte Carlo estimates of the expectation.

## Methodology

### Spectral Representation

The CAR model uses a graph Laplacian eigendecomposition:

```
φ ~ N(0, τ²(D - ρW)⁻¹) = N(0, τ²UΛ⁻¹U^T)
```

Where:
- `U` = eigenvectors of graph Laplacian
- `Λ` = diagonal matrix of eigenvalues
- Transform to spectral domain: `α = U^T φ`
- Each `α_j ~ N(0, τ²/λ_j)` independently

### Flexible Spectral Densities

Rather than fixed `τ²/λ_j`, we use Chebyshev polynomial approximation:

```
p(λ_j; θ) = exp(Σ_k θ_k T_k(λ̃_j))
```

Where:
- `T_k` = Chebyshev polynomial of order k
- `λ̃_j` = normalized eigenvalue ∈ [-1, 1]
- `θ` = learned spectral coefficients

This allows the model to learn arbitrary spatial smoothness patterns.

### Variational Inference

We approximate the posterior with a factorized distribution:

```
q(β, θ, τ², σ²) = q(β)q(θ)q(τ²)q(σ²)
```

- `q(β) = N(μ_β, Σ_β)` - Fixed effects
- `q(θ) = N(μ_θ, Σ_θ)` - Spectral coefficients  
- `q(τ²) = InverseGamma(a_τ, b_τ)` - Spatial variance
- `q(σ²) = InverseGamma(a_σ, b_σ)` - Observation noise

Optimize ELBO using Adam with Monte Carlo gradient estimates.

---

## API Reference

### Models

#### `SpectralCARBase`

Abstract base class for all spectral CAR models.

**Attributes:**
- `eigenvalues`: Graph Laplacian eigenvalues
- `eigenvectors`: Graph Laplacian eigenvectors
- `poly_order`: Order of Chebyshev polynomial approximation
- `mu_beta`, `mu_theta`: Variational means for parameters
- `a_tau`, `b_tau`, `a_sigma`, `b_sigma`: Inverse-Gamma parameters

**Methods:**
- `elbo(y, X)`: Compute evidence lower bound
- `sample_variational_params(n_samples)`: Sample from variational distribution

#### `SpectralCARMeanField(SpectralCARBase)`

Mean-field approximation with marginalized spatial effects.

**Parameters:**
- `n_obs`: Number of observations
- `n_features`: Number of fixed effect features
- `eigenvalues`: Graph Laplacian eigenvalues `(n_obs,)`
- `eigenvectors`: Graph Laplacian eigenvectors `(n_obs, n_obs)`
- `poly_order`: Chebyshev polynomial order (default: 5)
- `n_mc_samples`: Initial MC samples for ELBO (default: 20)
- `prior_beta_std`: Prior std for β (default: 5.0)
- `prior_theta_std`: Prior std for θ (default: 1.0)
- `prior_tau_a`, `prior_tau_b`: Gamma prior for τ² (default: 3.0, 1.0)
- `prior_sigma_a`, `prior_sigma_b`: Gamma prior for σ² (default: 5.0, 1.0)

### Inference

#### `VariationalInference`

Standard variational inference engine.

**Parameters:**
- `model`: SpectralCAR model instance

**Methods:**
- `fit(y, X, **kwargs)`: Fit model using VI
- `get_convergence_summary()`: Get convergence diagnostics

#### `CalibratedVI(VariationalInference)`

Variational inference with post-hoc calibration.

**Parameters:**
- `model`: SpectralCAR model instance
- `calibration_factor`: Pre-computed factor (optional)
- `target_coverage`: Target coverage probability (default: 0.95)

**Methods:**
- `fit(y_train, X_train, y_val, X_val, **kwargs)`: Fit with auto-calibration
- `calibrate(y_val, X_val)`: Calibrate on validation data
- `predict(y, X, calibrated=True)`: Get calibrated predictions

---

## Performance Tips

### For large datasets (>1000 locations):

1. Use `SpectralCARMeanField` for speed
2. Start with lower `poly_order` (3-5)
3. Use smaller initial `n_mc_samples` (5-10)
4. Increase `n_mc_samples_final` gradually (20-50)

### For better accuracy:

1. Use `SpectralCARCollapsed` or `SpectralCARLowRank`
2. Increase `poly_order` (6-10)
3. Use more `n_mc_samples_final` (50-100)
4. Run more iterations (2000-5000)

### For stable training:

1. Use learning rate scheduling (`use_scheduler=True`)
2. Enable gradient clipping (default: 5.0)
3. Use warmup period before ramping MC samples
4. Monitor ELBO - should increase steadily

---

## Troubleshooting

### Training instability (ELBO oscillates)

- Reduce `learning_rate` (try 0.005-0.01)
- Increase `warmup_iterations` (500-1000)
- Check for numerical issues (NaN/Inf in gradients)

### Poor spatial field recovery

- Increase `poly_order` (6-8)
- Check eigenvalue range (should span 0 to max)
- Verify graph connectivity
- Try different benchmark datasets to test

### Low coverage (<90%)

- Use `CalibratedVI` with validation data
- Increase `n_mc_samples_final` (50-100)
- Check if model is underestimating uncertainty

### Slow convergence

- Increase `learning_rate` (0.02-0.05)
- Reduce `n_mc_samples_initial` (3-5)
- Use smaller `poly_order` initially

---

## Citation

If you use this package in your research, please cite:

```bibtex
@software{spectral_car2025,
  author = {Sean Plummer},
  title = {Spectral CAR: Spectral Conditional Autoregressive Models},
  year = {2025},
  url = {https://github.com/scplummer/spectral-car}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

- Issues: https://github.com/scplummer/spectral-car/issues
- Email: seanp@uark.edu

---

## Acknowledgments

Based on research in:
- Spectral methods for spatial statistics
- Variational Bayes for hierarchical models
- Graph-based spatial modeling