"""
Example usage of train_models.py

This script demonstrates how to use the unified model training interface.
"""

import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from current directory (FREQ_NETS_TF)
from trainer import ModelTrainer, run_all_models
from models import create_model, ModelBuilder
from data_preprocessing import preprocess_pipeline
import pandas as pd


def example_1_single_model():
    """Example 1: Train a single model configuration."""
    print("="*60)
    print("Example 1: Training CNN with Business Hour features")
    print("="*60)
    
    # Load data (assuming it's already preprocessed and in 'data' directory)
    # If you need to download, use preprocess_pipeline with download_from_drive=True
    data = preprocess_pipeline(
        data_path="data",
        download_from_drive=False,
        years=[2016, 2017, 2018, 2019, 2020, 2021]
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model_type='cnn',
        feature_type='business_hour',
        window_size=2,
        n_forecast_steps=24,
        epochs=100,
        batch_size=168,
        learning_rate=0.003,
        patience=3,
        verbose=True
    )
    
    # Train model
    results = trainer.train(data, n_train=35064, save_model=True)
    
    # Access results
    print(f"\nFinal RMSE: {results['overall_rmse']:.2f}")
    print(f"Final MAE: {results['overall_mae']:.2f}")
    print(f"Final R²: {results['overall_r2']:.4f}")
    
    return results


def example_2_compare_features():
    """Example 2: Compare different feature combinations for the same model."""
    print("\n" + "="*60)
    print("Example 2: Comparing feature combinations for LSTM")
    print("="*60)
    
    # Load data
    data = preprocess_pipeline(
        data_path="data",
        download_from_drive=False
    )
    
    feature_types = ['ts_only', 'weekend', 'business_hour']
    results_comparison = []
    
    for feature_type in feature_types:
        print(f"\n--- Training LSTM with {feature_type} features ---")
        
        trainer = ModelTrainer(
            model_type='lstm',
            feature_type=feature_type,
            window_size=1,
            verbose=True
        )
        
        results = trainer.train(data, n_train=35064, save_model=False)
        
        results_comparison.append({
            'features': feature_type,
            'rmse': results['overall_rmse'],
            'mae': results['overall_mae'],
            'r2': results['overall_r2']
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_comparison)
    print("\n" + "="*60)
    print("Feature Comparison Results:")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def example_3_compare_models():
    """Example 3: Compare all models with best feature combination."""
    print("\n" + "="*60)
    print("Example 3: Comparing all models with business_hour features")
    print("="*60)
    
    # Load data
    data = preprocess_pipeline(
        data_path="data",
        download_from_drive=False
    )
    
    model_types = ['mlp', 'lstm', 'cnn']
    window_sizes = {'mlp': 24, 'lstm': 1, 'cnn': 2}
    results_comparison = []
    
    for model_type in model_types:
        print(f"\n--- Training {model_type.upper()} ---")
        
        trainer = ModelTrainer(
            model_type=model_type,
            feature_type='business_hour',
            window_size=window_sizes[model_type],
            verbose=True
        )
        
        results = trainer.train(data, n_train=35064, save_model=False)
        
        results_comparison.append({
            'model': model_type.upper(),
            'rmse': results['overall_rmse'],
            'mae': results['overall_mae'],
            'r2': results['overall_r2']
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_comparison)
    comparison_df = comparison_df.sort_values('rmse')
    
    print("\n" + "="*60)
    print("Model Comparison Results (sorted by RMSE):")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def example_4_run_all():
    """Example 4: Run all model configurations automatically."""
    print("\n" + "="*60)
    print("Example 4: Running all model configurations")
    print("="*60)
    
    # This will run all 9 combinations (3 models × 3 feature types)
    results_df = run_all_models(
        data_path="data",
        download_from_drive=False,
        n_train=35064,
        save_results=True,
        results_path="results"
    )
    
    print("\nBest model configuration:")
    print(results_df.iloc[0])
    
    return results_df


def example_5_custom_configuration():
    """Example 5: Custom configuration with different parameters."""
    print("\n" + "="*60)
    print("Example 5: Custom configuration")
    print("="*60)
    
    # Load data
    data = preprocess_pipeline(
        data_path="data",
        download_from_drive=False
    )
    
    # Custom trainer with different hyperparameters
    trainer = ModelTrainer(
        model_type='cnn',
        feature_type='business_hour',
        window_size=2,
        n_forecast_steps=24,  # Forecast 24 hours ahead
        epochs=50,  # Fewer epochs for faster training
        batch_size=128,  # Different batch size
        learning_rate=0.001,  # Lower learning rate
        patience=5,  # More patience for early stopping
        verbose=True
    )
    
    results = trainer.train(
        data,
        n_train=35064,
        save_model=True,
        model_path="custom_cnn_model.h5"
    )
    
    return results


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # Example 1: Single model
    # results = example_1_single_model()
    
    # Example 2: Compare features
    # comparison = example_2_compare_features()
    
    # Example 3: Compare models
    # comparison = example_3_compare_models()
    
    # Example 4: Run all configurations
    # results_df = example_4_run_all()
    
    # Example 5: Custom configuration
    # results = example_5_custom_configuration()
    
    print("\n" + "="*60)
    print("Examples ready! Uncomment the example you want to run.")
    print("="*60)
    print("\nTo run from command line:")
    print("  python train_models.py --model all --features all")
    print("\nTo run specific model:")
    print("  python train_models.py --model cnn --features business_hour")

