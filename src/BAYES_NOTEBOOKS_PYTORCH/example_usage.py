"""
Example usage of Bayesian neural network models for energy forecasting.

This script demonstrates how to use the BayesianModelTrainer to train
and evaluate Bayesian MLP models with uncertainty quantification.
"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trainer import BayesianModelTrainer, run_all_models
from data_preprocessing import preprocess_pipeline


def example_single_model():
    """Example: Train a single Bayesian MLP model."""
    print("="*60)
    print("Example: Training Single Bayesian MLP Model")
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
        device=None  # Auto-detect
    )
    
    # Train model
    results = trainer.train(data, n_train=35064, save_model=False)
    
    # Print results
    print(f"\nResults:")
    print(f"RMSE: {results['overall_rmse']:.2f}")
    print(f"MAE: {results['overall_mae']:.2f}")
    print(f"R²: {results['overall_r2']:.4f}")
    print(f"Coverage: {results['coverage']:.2%}")


def example_weekend_features():
    """Example: Train Bayesian MLP with weekend dummy variables."""
    print("="*60)
    print("Example: Bayesian MLP with Weekend Features")
    print("="*60)
    
    # Load data
    data = preprocess_pipeline(
        data_path="data",
        download_from_drive=False,
        years=[2019, 2020, 2021]
    )
    
    # Create trainer with weekend features
    trainer = BayesianModelTrainer(
        feature_type='weekend',
        window_size=1,
        n_forecast_steps=24,
        epochs=5,
        batch_size=168,
        learning_rate=0.01,
        verbose=True
    )
    
    # Train
    results = trainer.train(data, n_train=35064)
    
    print(f"\nResults:")
    print(f"RMSE: {results['overall_rmse']:.2f}")
    print(f"Coverage: {results['coverage']:.2%}")


def example_business_hour():
    """Example: Train Bayesian MLP with business hour dummy."""
    print("="*60)
    print("Example: Bayesian MLP with Business Hour Feature")
    print("="*60)
    
    # Load data
    data = preprocess_pipeline(
        data_path="data",
        download_from_drive=False,
        years=[2019, 2020, 2021]
    )
    
    # Create trainer
    trainer = BayesianModelTrainer(
        feature_type='business_hour',
        window_size=1,
        n_forecast_steps=24,
        epochs=5,
        batch_size=168,
        learning_rate=0.01,
        verbose=True
    )
    
    # Train
    results = trainer.train(data, n_train=35064)
    
    print(f"\nResults:")
    print(f"RMSE: {results['overall_rmse']:.2f}")
    print(f"Coverage: {results['coverage']:.2%}")


def example_run_all():
    """Example: Run all model configurations automatically."""
    print("="*60)
    print("Example: Running All Bayesian Model Configurations")
    print("="*60)
    
    results_df = run_all_models(
        data_path="data",
        download_from_drive=False,
        years=[2019, 2020, 2021],
        n_train=35064,
        save_results=True,
        results_path="results"
    )
    
    print("\nComparison Results:")
    print(results_df)


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # example_single_model()
    # example_weekend_features()
    # example_business_hour()
    example_run_all()

