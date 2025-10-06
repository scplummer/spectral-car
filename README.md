# Spectral CAR Models

Conditional Autoregressive (CAR) spatial models using spectral representations and variational inference.

## Overview

This package implements spatial models for areal data using spectral decomposition of graph Laplacians. Instead of working directly with large precision matrices, we parameterize spatial structure through the spectral density - a function that controls smoothness at different spatial frequencies.

### Key Features

- **Flexible spectral densities**: Choose between parametric forms (rational, exponential, power-law) or flexible polynomial bases (Chebyshev)
- **Efficient inference**: O(n) complexity per iteration using spectral decomposition
- **Multiple model variants**:
  - `SpectralCARMeanField`: Fast, marginalized spatial effects
  - `SpectralCARJoint`: Full uncertainty quantification for spatial predictions
  - `SpectralCARLowRank`: Captures posterior correlations with low-rank structure
- **Calibrated uncertainty**: Post-hoc calibration for accurate prediction intervals

## Installation

```bash
# Clone the repository
git clone https://github.com/scplummer/spectral-car.git
cd spectral-car

# Install in development mode
pip install -e .
```

## Quick Start

```python
import torch
from spectral_car.utils import create_grid_graph_laplacian, generate_benchmark_dataset
from spectral_car.models import SpectralCARMeanField
from spectral_car.inference import VariationalInference

# Create spatial structure (10x10 grid)
grid_size = 10
n_obs = grid_size ** 2
eigenvalues, eigenvectors = create_grid_graph_laplacian(n_obs, grid_size)

# Generate synthetic data
data = generate_benchmark_dataset(
    'smooth', 
    n_obs=n_obs,
    grid_size=grid_size,
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    seed=42
)

# Initialize model with parametric spectral form
model = SpectralCARMeanField(
    n_obs=n_obs,
    n_features=data['X'].shape[1],
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    spectral_form='rational',  # Simple 2-parameter form
    prior_beta_std=10.0,
    prior_theta_std=0.5
)

# Fit model
vi = VariationalInference(model=model)
history = vi.fit(
    y=data['y'],
    X=data['X'],
    n_iterations=2000,
    learning_rate=0.01
)

# Make predictions
phi_mean, phi_std = model.predict_spatial_effect(data['y'], data['X'])
```

## Spectral Density Forms

The spectral density `p(λ)` controls spatial smoothness. We support several parametric forms:

### Rational (recommended for most cases)
```python
spectral_form='rational'  # p(λ) = 1/(a + b*λ)
# Parameters: θ = [log(a), log(b)]
# - 2 parameters only
# - Natural 1/x decay
# - No oscillations
```

### Exponential
```python
spectral_form='exponential'  # p(λ) = exp(-a - b*λ)
# Parameters: θ = [log(a), log(b)]
# - 2 parameters
# - Exponential decay
# - Good for very smooth fields
```

### Power Law
```python
spectral_form='power_law'  # p(λ) = 1/(a + b*λ)^d
# Parameters: θ = [log(a), log(b), log(d)]
# - 3 parameters
# - Flexible decay rate
# - Generalizes rational form
```

### Chebyshev Polynomials
```python
spectral_form='chebyshev'  # p(λ) = exp(Σ θ_k T_k(λ))
# Parameters: θ = [θ_0, ..., θ_K]
# - K+1 parameters (typically 5-6)
# - Most flexible
# - Can oscillate for some functions
```

**Recommendation**: Start with `'rational'` - it's simple, stable, and captures typical CAR behavior well.

## Model Comparison

| Model | Spatial Effects | Uncertainty | Complexity | Use Case |
|-------|----------------|-------------|------------|----------|
| `SpectralCARMeanField` | Marginalized | Limited | O(n) | Fast hyperparameter estimation |
| `SpectralCARJoint` | Explicit | Full | O(n·S) | Spatial prediction with UQ |
| `SpectralCARLowRank` | Correlated | Full + correlations | O(n·k²) | When posterior correlations matter |

Where S = MC samples, k = low-rank dimension.

## Examples

### Basic Usage
```bash
python examples/basic_example.py
```

### Benchmark Comparison
```python
from spectral_car.utils import generate_benchmark_dataset

# Different spatial patterns
benchmarks = ['smooth', 'rough', 'multi_scale', 'anisotropic']

for name in benchmarks:
    data = generate_benchmark_dataset(name, n_obs, grid_size, 
                                      eigenvalues, eigenvectors)
    # Fit and evaluate...
```

### Calibrated Predictions
```python
from spectral_car.inference import CalibratedVI

# Fit with calibration
vi = CalibratedVI(model)
history = vi.fit(
    y_train, X_train,
    y_val, X_val,
    auto_calibrate=True  # Calibrate on validation data
)

# Get calibrated predictions
phi_mean, phi_lower, phi_upper = vi.get_credible_intervals(
    y_test, X_test, 
    confidence=0.95,
    calibrated=True
)
```

## Project Structure

```
spectral-car/
├── spectral_car/
│   ├── models.py           # SpectralCARBase and variants
│   ├── inference.py        # Variational inference engines
│   ├── utils/
│   │   ├── data.py         # Data generation utilities
│   │   ├── graph.py        # Graph Laplacian computation
│   │   ├── spectral.py     # Spectral transformation utilities
│   │   └── metrics.py      # Evaluation metrics
│   └── visualization.py    # Plotting functions
├── examples/
│   ├── basic_example.py    # Getting started
│   └── benchmark_study.py  # Model comparison
└── tests/
    ├── test_models.py
    ├── test_inference.py
    └── test_utils.py
```

## Mathematical Details

### Model Specification

The spatial model is:
```
y = Xβ + φ + ε
φ ~ CAR(θ, τ²)
ε ~ N(0, σ²I)
```

The CAR prior has precision matrix:
```
Q = τ² · U · diag(p(λ₁), ..., p(λₙ)) · U^T
```

where:
- `U` = eigenvectors of graph Laplacian
- `λᵢ` = eigenvalues of graph Laplacian
- `p(λ)` = spectral density function

### Why Parametric Forms?

Chebyshev polynomials are flexible but can:
- Oscillate heavily for non-smooth spectral densities
- Require many parameters (5-6+)
- Be numerically unstable at high orders

Parametric forms like rational `1/(a + b*λ)`:
- Match typical CAR behavior (1/x decay)
- Use only 2 parameters
- Never oscillate
- Are numerically stable

### Variational Inference

We use mean-field or structured variational approximations:
```
q(β, θ, τ², σ²) = q(β)q(θ)q(τ²)q(σ²)
```

For joint models, we additionally include:
```
q(α) = N(μ_α, Σ_α)  where φ = U·α
```

The ELBO is optimized using Adam with MC gradient estimates.

## Performance Tips

1. **Start simple**: Use `spectral_form='rational'` with `SpectralCARMeanField`
2. **Convergence**: Run 2000-5000 iterations for stable estimates
3. **Learning rates**: Use 0.01-0.02 for most parameters, 0.003 for variances
4. **MC samples**: Ramp from 20 to 80 samples during training
5. **Calibration**: Always use validation set for uncertainty calibration

## Citation

If you use this code, please cite:

```bibtex
@software{spectral_car,
  title = {Spectral CAR Models: Efficient Spatial Modeling with Parametric Spectral Densities},
  author = {Sean Plummer},
  year = {2025},
  url = {https://github.com/scplummer/spectral-car}
}
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

Key papers and resources:
- Rue & Held (2005): Gaussian Markov Random Fields
- Bradley et al. (2015): Computationally efficient spatial models
- Ver Hoef et al. (2018): Spatial autoregressive models