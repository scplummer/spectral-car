"""
Inference methods for spectral CAR models.

This module provides variational inference engines for fitting spectral CAR models,
including standard VI and calibrated VI with post-hoc uncertainty adjustment.
"""
from typing import List, Dict, Optional, Tuple

import torch
from scipy import stats

class VariationalInference:
    """
    Base class for variational inference on spectral CAR models.
    
    This class implements the core optimization loop for variational Bayes,
    including:
    - Adam optimization with learning rate scheduling
    - MC sample ramping during training
    - Progress tracking and diagnostics
    - Gradient clipping for stability
    
    Args:
        model: SpectralCAR model instance (any subclass of SpectralCARBase)
    
    Attributes:
        model: The model being fit
        history: List of dictionaries containing training diagnostics
        optimizer: PyTorch optimizer (created during fit)
        scheduler: Learning rate scheduler (created during fit if use_scheduler=True)
    """
    
    def __init__(self, model):
        self.model = model
        self.history = []
        self.optimizer = None
        self.scheduler = None
        
    def fit(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        n_iterations: int = 3000,
        learning_rate: float = 0.02,
        n_mc_samples_final: Optional[int] = None,
        warmup_iterations: int = 500,
        use_scheduler: bool = True,
        gradient_clip: float = 5.0,
        verbose: bool = True,
        print_every: int = 100
    ) -> List[Dict]:
        """
        Fit model using variational inference.
        
        Args:
            y: Observations (n_obs,)
            X: Design matrix (n_obs, n_features)
            n_iterations: Number of optimization iterations
            learning_rate: Initial learning rate
            n_mc_samples_final: Final number of MC samples (ramped up during training).
                              If None, uses model's n_mc_samples_initial
            warmup_iterations: Number of iterations before ramping up MC samples
            use_scheduler: Whether to use ReduceLROnPlateau scheduler
            gradient_clip: Maximum gradient norm for clipping
            verbose: Whether to print progress
            print_every: Print frequency (in iterations)
            
        Returns:
            history: List of dictionaries with training diagnostics
        """
        # Set final MC samples if not provided
        if n_mc_samples_final is None:
            n_mc_samples_final = self.model.n_mc_samples_initial * 4
        
        # Setup optimizer with different learning rates for different parameter groups
        self.optimizer = self._create_optimizer(learning_rate)
        
        # Setup learning rate scheduler
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max',  # Maximize ELBO
                factor=0.7, 
                patience=150,
                threshold=0.01,
                min_lr=1e-6
            )
        
        # Training loop
        self.history = []
        best_elbo = -float('inf')
        
        for iteration in range(n_iterations):
            # Gradually increase MC samples after warmup
            self._update_mc_samples(iteration, n_iterations, warmup_iterations, n_mc_samples_final)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute ELBO
            elbo_value, diagnostics = self.model.elbo(y, X)
            loss = -elbo_value  # Minimize negative ELBO
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Learning rate scheduling
            if use_scheduler and self.scheduler is not None:
                self.scheduler.step(elbo_value.detach())
            
            # Track best ELBO
            if elbo_value.item() > best_elbo:
                best_elbo = elbo_value.item()
            
            # Collect diagnostics
            diagnostics['elbo'] = elbo_value.item()
            diagnostics['iteration'] = iteration
            diagnostics['n_mc_samples'] = self.model.n_mc_samples
            diagnostics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            diagnostics['best_elbo'] = best_elbo
            
            # Add model-specific diagnostics
            with torch.no_grad():
                diagnostics.update(self._get_additional_diagnostics())
            
            self.history.append(diagnostics)
            
            # Print progress
            if verbose and (iteration % print_every == 0 or iteration == n_iterations - 1):
                self._print_progress(iteration, diagnostics)
        
        if verbose:
            print(f"\nTraining complete! Best ELBO: {best_elbo:.2f}")
        
        return self.history
    
    def _create_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        """
        Create optimizer with parameter-specific learning rates.
        
        Different parameter types may benefit from different learning rates:
        - Fixed effects (beta): standard rate
        - Spatial effects (alpha/phi): slightly higher rate
        - Spectral coefficients (theta): slightly lower rate
        - Variance parameters (tau^2, sigma^2): much lower rate (more stable)
        
        Args:
            learning_rate: Base learning rate
            
        Returns:
            Adam optimizer with parameter groups
        """
        # Collect parameters by type
        param_groups = []
        
        # Beta parameters
        if hasattr(self.model, 'mu_beta'):
            param_groups.append({
                'params': [self.model.mu_beta, self.model.log_diag_sigma_beta],
                'lr': learning_rate
            })
        
        # Alpha parameters (if joint or low-rank model)
        if hasattr(self.model, 'mu_alpha'):
            alpha_params = [self.model.mu_alpha, self.model.log_sigma_alpha]
            # Add low-rank factors if present
            if hasattr(self.model, 'L_alpha'):
                alpha_params.append(self.model.L_alpha)
            param_groups.append({
                'params': alpha_params,
                'lr': learning_rate * 1.0
            })
        
        # Theta parameters
        if hasattr(self.model, 'mu_theta'):
            param_groups.append({
                'params': [self.model.mu_theta, self.model.log_diag_sigma_theta],
                'lr': learning_rate * 0.8
            })
        
        # Variance parameters (tau^2, sigma^2) - slower learning rate
        variance_params = []
        if hasattr(self.model, 'log_a_tau'):
            variance_params.extend([self.model.log_a_tau, self.model.log_b_tau])
        if hasattr(self.model, 'log_a_sigma'):
            variance_params.extend([self.model.log_a_sigma, self.model.log_b_sigma])
        
        if variance_params:
            param_groups.append({
                'params': variance_params,
                'lr': learning_rate * 0.3
            })
        
        return torch.optim.Adam(param_groups, lr=learning_rate)
    
    def _update_mc_samples(
        self, 
        iteration: int,
        n_iterations: int, 
        warmup_iterations: int, 
        n_mc_samples_final: int
    ):
        """
        Gradually increase MC samples during training.
    
        Start with fewer samples for speed, then increase for accuracy.
        """
        if iteration < warmup_iterations:
            self.model.n_mc_samples = self.model.n_mc_samples_initial
        else:
            # Compute progress from 0 to 1 after warmup
            progress = (iteration - warmup_iterations) / max(1, n_iterations - warmup_iterations)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
        
            # Linearly interpolate between initial and final
            self.model.n_mc_samples = int(
                self.model.n_mc_samples_initial + 
                progress * (n_mc_samples_final - self.model.n_mc_samples_initial)
            )
        
            # Ensure at least 1 sample
            self.model.n_mc_samples = max(1, self.model.n_mc_samples)
    
    def _get_additional_diagnostics(self) -> dict:
        """
        Get model-specific diagnostics (e.g., current parameter values).
        
        Can be overridden by subclasses for custom diagnostics.
        
        Returns:
            Dictionary of diagnostic values
        """
        diagnostics = {}
        
        # Variance parameters
        if hasattr(self.model, 'a_tau'):
            tau2_mean = (self.model.b_tau / (self.model.a_tau - 1)).item()
            diagnostics['tau2_current'] = tau2_mean
        
        if hasattr(self.model, 'a_sigma'):
            sigma2_mean = (self.model.b_sigma / (self.model.a_sigma - 1)).item()
            diagnostics['sigma2_current'] = sigma2_mean
        
        # Low-rank contribution
        if hasattr(self.model, 'L_alpha'):
            diagnostics['lowrank_norm'] = torch.norm(self.model.L_alpha).item()
            diagnostics['sigma_alpha_mean'] = torch.mean(self.model.sigma_alpha).item()
            diagnostics['sigma_alpha_max'] = torch.max(self.model.sigma_alpha).item()
        
        return diagnostics
    
    def _print_progress(self, iteration: int, diagnostics: dict):
        """
        Print training progress.
        
        Args:
            iteration: Current iteration
            diagnostics: Dictionary of diagnostic values
        """
        # Base info
        print(f"Iter {iteration:4d} | ELBO: {diagnostics['elbo']:8.2f} | ", end="")
        
        # Expected log likelihood
        if 'expected_log_lik' in diagnostics:
            print(f"E[log p(y|.)]: {diagnostics['expected_log_lik']:8.2f} | ", end="")
        
        # For joint models, show entropy and prior
        if 'entropy_spatial' in diagnostics:
            print(f"H[q(φ)]: {diagnostics['entropy_spatial']:6.2f} | ", end="")
        
        # KL divergence
        if 'total_kl' in diagnostics:
            print(f"KL: {diagnostics['total_kl']:6.2f} | ", end="")
        elif 'total_kl_hyperparams' in diagnostics:
            print(f"KL: {diagnostics['total_kl_hyperparams']:6.2f} | ", end="")
        
        # Low-rank info
        if 'lowrank_norm' in diagnostics:
            print(f"||L||: {diagnostics['lowrank_norm']:5.2f} | ", end="")
            print(f"σ_max: {diagnostics['sigma_alpha_max']:5.2f} | ", end="")
        
        # MC samples
        print(f"MC: {diagnostics['n_mc_samples']:2d}")
    
    def get_convergence_summary(self) -> dict:
        """
        Get summary of convergence diagnostics.
        
        Returns:
            Dictionary with convergence metrics
        """
        if not self.history:
            return {}
        
        elbos = [h['elbo'] for h in self.history]
        
        return {
            'final_elbo': elbos[-1],
            'best_elbo': max(elbos),
            'elbo_improvement': elbos[-1] - elbos[0],
            'n_iterations': len(elbos),
            'final_learning_rate': self.history[-1]['learning_rate'],
            'converged': self._check_convergence(elbos),
        }
    
    def _check_convergence(self, elbos: List[float], window: int = 100, threshold: float = 0.1) -> bool:
        """
        Check if ELBO has converged.
        
        Args:
            elbos: List of ELBO values
            window: Number of recent iterations to check
            threshold: Convergence threshold (change in ELBO)
            
        Returns:
            True if converged, False otherwise
        """
        if len(elbos) < window:
            return False
        
        recent_elbos = elbos[-window:]
        elbo_change = abs(recent_elbos[-1] - recent_elbos[0])
        
        return elbo_change < threshold

class CalibratedVI(VariationalInference):
    """
    Variational inference with empirical calibration for uncertainty quantification.
    
    This class extends standard VI by adding post-hoc calibration of posterior
    uncertainties. Variational inference often underestimates or overestimates
    uncertainties due to mean-field approximations. This class learns a calibration
    factor from validation data to adjust uncertainties to achieve nominal coverage.
    
    The calibration process:
    1. Fit model on training data using standard VI
    2. Predict on validation data
    3. Find scaling factor that achieves target coverage (e.g., 95%)
    4. Apply factor to all future predictions
    
    Args:
        model: SpectralCAR model instance
        calibration_factor: Pre-computed calibration factor (optional).
                          If None, must call calibrate() before calibrated predictions.
        target_coverage: Target coverage probability for calibration (default: 0.95)
    
    Attributes:
        calibration_factor: Current calibration factor (multiplies std devs)
        target_coverage: Target coverage probability
        calibration_diagnostics: Diagnostics from calibration process
    
    Example:
        >>> model = SpectralCARJoint(...)
        >>> vi = CalibratedVI(model)
        >>> 
        >>> # Fit with automatic calibration
        >>> vi.fit(y_train, X_train, y_val, X_val)
        >>> 
        >>> # Get calibrated predictions
        >>> phi_mean, phi_std = vi.predict(y_test, X_test, calibrated=True)
    """
    
    def __init__(
        self, 
        model, 
        calibration_factor: Optional[float] = None,
        target_coverage: float = 0.95
    ):
        super().__init__(model)
        self.calibration_factor = calibration_factor
        self.target_coverage = target_coverage
        self.calibration_diagnostics = {}
        
    def fit(
        self,
        y_train: torch.Tensor,
        X_train: torch.Tensor,
        y_val: Optional[torch.Tensor] = None,
        X_val: Optional[torch.Tensor] = None,
        auto_calibrate: bool = True,
        **kwargs
    ) -> List[Dict]:
        """
        Fit model with optional automatic calibration on validation data.
        
        Args:
            y_train: Training observations (n_train,)
            X_train: Training design matrix (n_train, n_features)
            y_val: Validation observations (n_val,) [optional]
            X_val: Validation design matrix (n_val, n_features) [optional]
            auto_calibrate: If True and validation data provided, auto-calibrate
            **kwargs: Additional arguments passed to VariationalInference.fit()
            
        Returns:
            history: Training history
        """
        # Fit using parent class
        history = super().fit(y_train, X_train, **kwargs)
        
        # Auto-calibrate if validation data provided
        if auto_calibrate and y_val is not None and X_val is not None:
            if kwargs.get('verbose', True):
                print("\nCalibrating uncertainties on validation data...")
            self.calibrate(y_val, X_val)
        
        return history
    
    def calibrate(
        self,
        y_val: torch.Tensor,
        X_val: torch.Tensor,
        target_coverage: Optional[float] = None,
        z_score: float = 1.96,
        method: str = 'binary_search',
        verbose: bool = True
    ):
        """
        Find calibration factor to achieve target coverage on validation data.
        
        Args:
            y_val: Validation observations (n_val,)
            X_val: Validation design matrix (n_val, n_features)
            target_coverage: Target coverage probability (uses self.target_coverage if None)
            z_score: Z-score for confidence intervals (1.96 for 95%, 1.0 for 68%)
            method: Calibration method ('binary_search' or 'analytical')
            verbose: Whether to print calibration results
        """
        if target_coverage is None:
            target_coverage = self.target_coverage
        
        with torch.no_grad():
            # Get predictions
            phi_mean, phi_std_raw = self._get_spatial_predictions(y_val, X_val)
            
            # Compute prediction errors
            # For validation data, we approximate true phi from residuals
            beta_mean = self.model.mu_beta
            residuals = y_val - X_val @ beta_mean
            phi_true_approx = residuals  # Approximate (includes observation noise)
            
            # Find calibration factor
            if method == 'binary_search':
                self.calibration_factor = self._binary_search_calibration(
                    phi_true_approx, phi_mean, phi_std_raw, target_coverage, z_score
                )
            elif method == 'analytical':
                self.calibration_factor = self._analytical_calibration(
                    phi_true_approx, phi_mean, phi_std_raw, target_coverage, z_score
                )
            else:
                raise ValueError(f"Unknown calibration method: {method}")
            
            # Compute diagnostics
            self._compute_calibration_diagnostics(
                phi_true_approx, phi_mean, phi_std_raw, z_score
            )
        
        if verbose:
            self._print_calibration_summary()
    
    def _binary_search_calibration(
        self,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        target_coverage: float,
        z_score: float
    ) -> float:
        """
        Find calibration factor via binary search.
        
        Searches for factor f such that:
        P(|true - pred| < f * z * std) = target_coverage
        
        Args:
            true_values: True/approximate values (n,)
            predictions: Predicted means (n,)
            uncertainties: Predicted std devs (n,)
            target_coverage: Target coverage probability
            z_score: Z-score for intervals
            
        Returns:
            Calibration factor
        """
        errors = torch.abs(true_values - predictions)
        
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
    
    def _analytical_calibration(
        self,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        target_coverage: float,
        z_score: float
    ) -> float:
        """
        Find calibration factor using analytical approximation.
        
        Assumes errors are approximately Gaussian and estimates the
        calibration factor from the empirical error distribution.
        
        Args:
            true_values: True/approximate values (n,)
            predictions: Predicted means (n,)
            uncertainties: Predicted std devs (n,)
            target_coverage: Target coverage probability
            z_score: Z-score for intervals
            
        Returns:
            Calibration factor
        """
        # Compute standardized errors: (true - pred) / std
        standardized_errors = (true_values - predictions) / uncertainties
        
        # Estimate empirical std of standardized errors
        empirical_std = torch.std(standardized_errors).item()
        
        # Calibration factor is ratio of empirical to theoretical std
        # For well-calibrated predictions, standardized errors should have std ≈ 1
        calibration_factor = empirical_std
        
        # Adjust if needed to achieve target coverage
        # (This is a simple heuristic; binary search is more accurate)
        return max(0.1, min(3.0, calibration_factor))
    
    def _compute_calibration_diagnostics(
        self,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
        uncertainties_raw: torch.Tensor,
        z_score: float
    ):
        """Compute diagnostics for calibration quality."""
        uncertainties_calibrated = uncertainties_raw * self.calibration_factor
        
        # Coverage at different levels
        errors = torch.abs(true_values - predictions)
        
        coverages = {}
        for level, z in [('68%', 1.0), ('95%', 1.96), ('99%', 2.58)]:
            # Raw coverage
            intervals_raw = z * uncertainties_raw
            cov_raw = torch.mean((errors <= intervals_raw).float()).item()
            
            # Calibrated coverage
            intervals_cal = z * uncertainties_calibrated
            cov_cal = torch.mean((errors <= intervals_cal).float()).item()
            
            coverages[level] = {
                'raw': cov_raw,
                'calibrated': cov_cal,
                'target': self._z_to_coverage(z)
            }
        
        # Error statistics
        self.calibration_diagnostics = {
            'calibration_factor': self.calibration_factor,
            'coverages': coverages,
            'mae': torch.mean(errors).item(),
            'rmse': torch.sqrt(torch.mean(errors**2)).item(),
            'n_validation': len(true_values)
        }
    
    def _z_to_coverage(self, z: float) -> float:
        """Convert z-score to coverage probability (approximate)."""
        import math
        from scipy import special
        try:
            # Use scipy if available
            return float(special.erf(z / math.sqrt(2)))
        except:
            # Fallback approximations
            if abs(z - 1.0) < 0.1:
                return 0.68
            elif abs(z - 1.96) < 0.1:
                return 0.95
            elif abs(z - 2.58) < 0.1:
                return 0.99
            else:
                return 0.95  # Default
    
    def _print_calibration_summary(self):
        """Print calibration results."""
        print(f"\n{'='*60}")
        print(f"CALIBRATION SUMMARY")
        print(f"{'='*60}")
        print(f"Calibration factor: {self.calibration_factor:.3f}")
        print(f"Validation set size: {self.calibration_diagnostics['n_validation']}")
        print(f"\nCoverage (target vs actual):")
        
        for level, cov in self.calibration_diagnostics['coverages'].items():
            print(f"  {level}: target={cov['target']:.1%}, "
                  f"raw={cov['raw']:.1%}, "
                  f"calibrated={cov['calibrated']:.1%}")
        
        print(f"\nPrediction errors:")
        print(f"  MAE: {self.calibration_diagnostics['mae']:.3f}")
        print(f"  RMSE: {self.calibration_diagnostics['rmse']:.3f}")
        print(f"{'='*60}\n")
    
    def _get_spatial_predictions(
        self,
        y: torch.Tensor,
        X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get spatial effect predictions from the model.
        
        Handles both models with explicit spatial effects (Joint, LowRank)
        and marginalized models (MeanField).
        """
        # Check if model has direct spatial prediction method
        if hasattr(self.model, 'get_spatial_effect_posterior'):
            return self.model.get_spatial_effect_posterior()
        elif hasattr(self.model, 'predict_spatial_effect'):
            return self.model.predict_spatial_effect(y, X)
        else:
            raise ValueError("Model does not support spatial effect prediction")
    
    def predict(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        calibrated: bool = True,
        return_phi: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with optionally calibrated uncertainties.
        
        Args:
            y: Observations (n_obs,)
            X: Design matrix (n_obs, n_features)
            calibrated: Whether to apply calibration factor
            return_phi: If True, return spatial effects; if False, return fitted values
            
        Returns:
            predictions: Predicted values (n_obs,)
            uncertainties: Standard deviations (n_obs,)
        """
        with torch.no_grad():
            if return_phi:
                # Predict spatial effects
                phi_mean, phi_std_raw = self._get_spatial_predictions(y, X)
                predictions = phi_mean
                uncertainties_raw = phi_std_raw
            else:
                # Predict fitted values: Xβ + φ
                phi_mean, phi_std_raw = self._get_spatial_predictions(y, X)
                beta_mean = self.model.mu_beta
                predictions = X @ beta_mean + phi_mean
                uncertainties_raw = phi_std_raw  # Ignoring β uncertainty for simplicity
            
            # Apply calibration if requested
            if calibrated and self.calibration_factor is not None:
                uncertainties = uncertainties_raw * self.calibration_factor
            else:
                uncertainties = uncertainties_raw
            
            return predictions, uncertainties
    
    def get_credible_intervals(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        confidence: float = 0.95,
        calibrated: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get credible intervals for predictions.
        
        Args:
            y: Observations (n_obs,)
            X: Design matrix (n_obs, n_features)
            confidence: Confidence level (e.g., 0.95 for 95% intervals)
            calibrated: Whether to use calibrated uncertainties
            
        Returns:
            mean: Point predictions (n_obs,)
            lower: Lower bounds (n_obs,)
            upper: Upper bounds (n_obs,)
        """
        # Get z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        predictions, uncertainties = self.predict(y, X, calibrated=calibrated)
        
        lower = predictions - z_score * uncertainties
        upper = predictions + z_score * uncertainties
        
        return predictions, lower, upper