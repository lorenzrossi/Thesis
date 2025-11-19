"""
Example usage of manual BayesByBackprop implementation for energy forecasting.

This script demonstrates how to use the manual BayesByBackprop implementation
(models_bayesbybackprop.py and trainer_bayesbybackprop.py) to train and evaluate
Bayesian MLP models with full transparency and control.
"""

import os
import sys

# Add parent directory to path for data_preprocessing
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from trainer_bayesbybackprop import BayesianModelTrainer
from models_bayesbybackprop import BayesianMLP, compute_elbo_loss, BayesianLinear
from data_preprocessing import preprocess_pipeline
import torch
import torch.nn as nn


def example_single_model():
    """Example: Train a single Bayesian MLP model using manual BayesByBackprop."""
    print("="*60)
    print("Example: Training Single Bayesian MLP (Manual BayesByBackprop)")
    print("="*60)
    
    # Load and preprocess data
    data = preprocess_pipeline(
        data_path="data",
        download_from_drive=False,
        years=[2019, 2020, 2021]
    )
    
    # Create trainer
    trainer = BayesianModelTrainer(
        feature_type='ts_only',
        window_size=1,  # Use lag_1
        n_forecast_steps=24,
        epochs=5,
        batch_size=168,
        learning_rate=0.01,
        verbose=True,
        device=None,  # Auto-detect
        n_samples=5,  # Monte Carlo samples for ELBO
        inference_samples=100  # Samples for prediction
    )
    
    # Train model
    results = trainer.train(data, n_train=35064, save_model=False)
    
    # Print results
    print(f"\nResults:")
    print(f"RMSE: {results['overall_rmse']:.2f}")
    print(f"MAE: {results['overall_mae']:.2f}")
    print(f"R²: {results['overall_r2']:.4f}")
    print(f"Coverage: {results['coverage']:.2%}")


def example_understanding_components():
    """Example: Understanding the components of BayesByBackprop."""
    print("="*60)
    print("Example: Understanding BayesByBackprop Components")
    print("="*60)
    
    # Create a simple Bayesian layer
    layer = BayesianLinear(
        in_features=5,
        out_features=1,
        prior_mu=0.0,
        prior_sigma=1.0,
        posterior_mu_init=0.0,
        posterior_rho_init=-3.0
    )
    
    print("\n1. BayesianLinear Layer Structure:")
    print(f"   Weight mu shape: {layer.weight_mu.shape}")
    print(f"   Weight rho shape: {layer.weight_rho.shape}")
    print(f"   Bias mu shape: {layer.bias_mu.shape}")
    print(f"   Bias rho shape: {layer.bias_rho.shape}")
    
    # Forward pass (samples weights)
    x = torch.randn(10, 5)
    y = layer(x)
    print(f"\n2. Forward Pass:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Note: Each forward pass samples new weights!")
    
    # KL divergence
    kl = layer.kl_divergence()
    print(f"\n3. KL Divergence:")
    print(f"   KL(q(w|θ) || p(w)) = {kl.item():.4f}")
    print(f"   This measures how different the learned distribution is from the prior")
    
    # Create a model
    model = BayesianMLP(
        input_dim=5,
        output_dim=1,
        hidden_units=10,
        n_layers=2
    )
    
    print(f"\n4. Model Structure:")
    print(f"   Number of Bayesian layers: {len(model.bayesian_layers)}")
    print(f"   Total KL divergence: {model.kl_divergence().item():.4f}")
    
    # ELBO calculation
    labels = torch.randn(10, 1)
    criterion = nn.MSELoss()
    elbo, likelihood, kl_div = compute_elbo_loss(
        model, x, labels, criterion,
        n_samples=5,
        complexity_cost_weight=1.0 / 10
    )
    
    print(f"\n5. ELBO Loss Components:")
    print(f"   ELBO = Likelihood + Complexity_Weight * KL")
    print(f"   ELBO = {elbo.item():.4f}")
    print(f"   Likelihood (MSE): {likelihood.item():.4f}")
    print(f"   KL Divergence: {kl_div.item():.4f}")
    print(f"   Complexity Weight: {1.0 / 10:.4f}")


def example_custom_parameters():
    """Example: Using custom prior parameters."""
    print("="*60)
    print("Example: Custom Prior Parameters")
    print("="*60)
    
    # Load data
    data = preprocess_pipeline(
        data_path="data",
        download_from_drive=False,
        years=[2019, 2020, 2021]
    )
    
    # Create trainer with custom prior parameters
    trainer = BayesianModelTrainer(
        feature_type='ts_only',
        window_size=1,
        n_forecast_steps=24,
        epochs=5,
        batch_size=168,
        learning_rate=0.01,
        verbose=True,
        # Custom model parameters
        prior_mu=0.0,           # Zero-mean prior
        prior_sigma=0.5,        # Tighter prior (smaller variance)
        posterior_mu_init=0.0,  # Initialize near zero
        posterior_rho_init=-2.0 # Different initial variance
    )
    
    # Train
    results = trainer.train(data, n_train=35064)
    
    print(f"\nResults with custom priors:")
    print(f"RMSE: {results['overall_rmse']:.2f}")
    print(f"Coverage: {results['coverage']:.2%}")


def example_comparison_blitz_vs_manual():
    """Example: Compare results between blitz and manual implementation."""
    print("="*60)
    print("Example: Comparing Blitz vs Manual Implementation")
    print("="*60)
    
    # Load data
    data = preprocess_pipeline(
        data_path="data",
        download_from_drive=False,
        years=[2019, 2020, 2021]
    )
    
    # Manual implementation
    print("\n1. Training with Manual BayesByBackprop:")
    trainer_manual = BayesianModelTrainer(
        feature_type='ts_only',
        window_size=1,
        epochs=5,
        verbose=True
    )
    results_manual = trainer_manual.train(data, n_train=35064)
    
    print(f"\nManual Implementation Results:")
    print(f"  RMSE: {results_manual['overall_rmse']:.2f}")
    print(f"  Coverage: {results_manual['coverage']:.2%}")
    
    # Note: To compare with blitz, you would need to import from trainer.py
    # and run the same experiment
    print("\n2. To compare with blitz implementation:")
    print("   from trainer import BayesianModelTrainer as BlitzTrainer")
    print("   trainer_blitz = BlitzTrainer(...)")
    print("   results_blitz = trainer_blitz.train(...)")


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # example_single_model()
    example_understanding_components()
    # example_custom_parameters()
    # example_comparison_blitz_vs_manual()

