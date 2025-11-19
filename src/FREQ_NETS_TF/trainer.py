"""
Training and testing module for energy forecasting models.

This module handles data preparation, rolling window training, and evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
from math import sqrt
from typing import Dict, List, Tuple, Optional, Literal
import warnings

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path to import preprocessing module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from data_preprocessing import preprocess_pipeline, retrieve_data, handle_missing_values, data_and_aggregator, businesshour_and_we_generation
except ImportError:
    # Fallback: try importing from parent directory
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "data_preprocessing", 
        os.path.join(parent_dir, "data_preprocessing.py")
    )
    data_preprocessing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_preprocessing)
    preprocess_pipeline = data_preprocessing.preprocess_pipeline
    retrieve_data = data_preprocessing.retrieve_data
    handle_missing_values = data_preprocessing.handle_missing_values
    data_and_aggregator = data_preprocessing.data_and_aggregator
    businesshour_and_we_generation = data_preprocessing.businesshour_and_we_generation

# Import model builder
from models import create_model, ModelBuilder, DEFAULT_CONFIGS


class DataPreparator:
    """Handles data preparation for different model types."""
    
    def __init__(
        self,
        model_type: Literal['mlp', 'lstm', 'cnn'],
        feature_type: Literal['ts_only', 'weekend', 'business_hour'],
        window_size: int = 24
    ):
        """
        Initialize data preparator.
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'mlp', 'lstm', or 'cnn'
        feature_type : str
            Feature combination: 'ts_only', 'weekend', or 'business_hour'
        window_size : int
            Size of input window (for LSTM/CNN) or number of lags (for MLP)
        """
        self.model_type = model_type
        self.feature_type = feature_type
        self.window_size = window_size
        self.scaler = None
        self.feature_names = []
    
    def _create_sliding_window(self, sequence: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences from time series data."""
        X, y = [], []
        for i in range(window_size, len(sequence)):
            X.append(sequence[i-window_size:i])
            y.append(sequence[i])
        return np.array(X), np.array(y)
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features based on feature_type.
        
        Returns:
        --------
        feature_array : np.ndarray
            Feature array with shape (n_samples, n_features)
        feature_names : List[str]
            Names of features used
        """
        if self.feature_type == 'ts_only':
            features = ['total_aggregated']
        elif self.feature_type == 'weekend':
            features = ['total_aggregated', 'saturday', 'sunday']
        elif self.feature_type == 'business_hour':
            features = ['total_aggregated', 'business hour']
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")
        
        feature_array = data[features].values
        self.feature_names = features
        
        return feature_array, self.feature_names
    
    def scale_features(self, feature_array: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features. Only scales the first column (total_aggregated), 
        leaves dummy variables unchanged.
        """
        scaled_array = np.copy(feature_array)
        
        if fit:
            self.scaler = StandardScaler()
            self.scaler.fit(feature_array[:, 0].reshape(-1, 1))
        
        # Scale only the first column (total_aggregated)
        scaled_array[:, 0] = self.scaler.transform(feature_array[:, 0].reshape(-1, 1)).flatten()
        
        return scaled_array
    
    def prepare_data_for_model(
        self, 
        train_data: np.ndarray, 
        test_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training based on model type.
        
        Returns:
        --------
        X_train, y_train, X_test, y_test
        """
        if self.model_type == 'mlp':
            # For MLP, use lag features
            df_train = pd.DataFrame(train_data, columns=[f'feat_{i}' for i in range(train_data.shape[1])])
            df_test = pd.DataFrame(test_data, columns=[f'feat_{i}' for i in range(test_data.shape[1])])
            
            # Create lags for the first column (total_aggregated)
            for i in range(1, self.window_size + 1):
                df_train[f'lag_{i}'] = df_train.iloc[:, 0].shift(i)
                df_test[f'lag_{i}'] = df_test.iloc[:, 0].shift(i)
            
            # Add other features (dummies) if present
            if train_data.shape[1] > 1:
                for i in range(1, train_data.shape[1]):
                    df_train[f'feat_{i}'] = train_data[:, i]
                    df_test[f'feat_{i}'] = test_data[:, i]
            
            # Drop NaN rows
            df_train = df_train.dropna()
            df_test = df_test.dropna()
            
            # Prepare X and y
            lag_cols = [col for col in df_train.columns if col.startswith('lag_')]
            other_cols = [col for col in df_train.columns if col.startswith('feat_') and col != 'feat_0']
            
            X_train = df_train[lag_cols + other_cols].values
            y_train = df_train['feat_0'].values
            
            X_test = df_test[lag_cols + other_cols].values
            y_test = df_test['feat_0'].values
            
            # Handle case where test data doesn't have enough history
            if len(X_test) == 0:
                X_test = np.array([df_train[lag_cols + other_cols].iloc[-1].values])
                y_test = np.array([df_test['feat_0'].iloc[0] if len(df_test) > 0 else df_train['feat_0'].iloc[-1]])
            
        elif self.model_type in ['lstm', 'cnn']:
            # For LSTM/CNN, use sliding window
            X_train, y_train = self._create_sliding_window(train_data, self.window_size)
            X_test, y_test = self._create_sliding_window(test_data, self.window_size)
            
            # Extract target (first column) for y
            y_train = y_train[:, 0] if y_train.ndim > 1 else y_train
            y_test = y_test[:, 0] if y_test.ndim > 1 else y_test
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        return X_train, y_train, X_test, y_test


class ModelTrainer:
    """Handles training and evaluation of forecasting models with rolling window approach."""
    
    def __init__(
        self,
        model_type: Literal['mlp', 'lstm', 'cnn'],
        feature_type: Literal['ts_only', 'weekend', 'business_hour'],
        window_size: int = 24,
        n_forecast_steps: int = 24,
        epochs: int = 100,
        batch_size: int = 168,
        learning_rate: float = 0.003,
        patience: int = 3,
        verbose: bool = True,
        **model_kwargs
    ):
        """
        Initialize model trainer.
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'mlp', 'lstm', or 'cnn'
        feature_type : str
            Feature combination: 'ts_only', 'weekend', or 'business_hour'
        window_size : int
            Size of input window (for LSTM/CNN) or number of lags (for MLP)
        n_forecast_steps : int
            Number of steps to forecast ahead
        epochs : int
            Number of training epochs per window
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        patience : int
            Early stopping patience
        verbose : bool
            Whether to print progress
        **model_kwargs
            Additional arguments for model creation
        """
        self.model_type = model_type
        self.feature_type = feature_type
        self.window_size = window_size
        self.n_forecast_steps = n_forecast_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.verbose = verbose
        self.model_kwargs = model_kwargs
        
        self.model = None
        self.data_preparator = DataPreparator(model_type, feature_type, window_size)
        self.results = {}
        
    def train(
        self, 
        data: pd.DataFrame,
        n_train: int = 35064,  # 3 years of hourly data
        save_model: bool = False,
        model_path: Optional[str] = None
    ) -> Dict:
        """
        Train model using rolling window approach.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed data with datetime index
        n_train : int
            Number of training samples
        save_model : bool
            Whether to save the final model
        model_path : str, optional
            Path to save model
        
        Returns:
        --------
        Dict with training results and metrics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training {self.model_type.upper()} model with {self.feature_type} features")
            print(f"{'='*60}")
        
        # Prepare features
        feature_array, feature_names = self.data_preparator.prepare_features(data)
        
        # Split data
        train_data_raw = feature_array[:n_train]
        test_data_raw = feature_array[n_train:]
        
        # Scale features
        train_data = self.data_preparator.scale_features(train_data_raw, fit=True)
        test_data = self.data_preparator.scale_features(test_data_raw, fit=False)
        
        # Prepare data for model
        X_train, y_train, X_test, y_test = self.data_preparator.prepare_data_for_model(
            train_data, test_data
        )
        
        # Get truth values (unscaled) for evaluation
        truth = test_data_raw[-len(y_test):, 0] if len(y_test) > 0 else test_data_raw[:, 0]
        
        # Create model
        tf.keras.backend.clear_session()
        
        if self.model_type == 'mlp':
            input_shape = (X_train.shape[1],)
        elif self.model_type in ['lstm', 'cnn']:
            input_shape = (X_train.shape[1], X_train.shape[2]) if X_train.ndim > 2 else (X_train.shape[1], 1)
        
        # Merge default config with user kwargs
        config = DEFAULT_CONFIGS[self.model_type].copy()
        config.update(self.model_kwargs)
        config['learning_rate'] = self.learning_rate
        
        self.model = create_model(self.model_type, input_shape, **config)
        
        if self.verbose:
            print(f"\nModel Architecture:")
            self.model.summary()
        
        # Rolling window training
        history_X = X_train.copy()
        history_y = y_train.copy()
        
        predictions = []
        period_errors = []
        period = 1
        
        # Adjust test data iteration based on available data
        test_iterations = range(0, len(X_test), self.n_forecast_steps)
        
        if len(list(test_iterations)) == 0:
            warnings.warn("No test data available for rolling window training. Using train data for evaluation.")
            test_iterations = [0]
            X_test = X_train[-self.n_forecast_steps:] if len(X_train) >= self.n_forecast_steps else X_train
            y_test = y_train[-self.n_forecast_steps:] if len(y_train) >= self.n_forecast_steps else y_train
            truth = train_data_raw[-len(y_test):, 0] if len(y_test) > 0 else train_data_raw[:, 0]
        
        for i in test_iterations:
            t_end = min(i + self.n_forecast_steps, len(X_test))
            X_test_block = X_test[i:t_end]
            y_test_block = y_test[i:t_end]
            truth_block = truth[i:t_end] if len(truth) > i else truth[-len(y_test_block):]
            
            if self.verbose:
                print(f"\nTraining and predicting for period {period} (samples {i} to {t_end})")
            
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=0
            )
            
            # Train model
            self.model.fit(
                history_X,
                history_y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                callbacks=[early_stopping]
            )
            
            # Make predictions
            yhat = self.model.predict(X_test_block, verbose=0)
            
            # Inverse transform predictions
            yhat_rescaled = self.data_preparator.scaler.inverse_transform(
                yhat.reshape(-1, 1)
            ).flatten()
            
            # Calculate period RMSE (fixed bug: proper parentheses)
            period_rmse = sqrt(np.mean((truth_block - yhat_rescaled)**2))
            period_errors.append(period_rmse)
            predictions.extend(yhat_rescaled)
            
            if self.verbose:
                print(f"Period RMSE: {period_rmse:.2f}")
            
            # Update history with actual observed values
            history_X = np.concatenate((history_X, X_test_block), axis=0)
            history_X = history_X[self.n_forecast_steps:]
            history_y = np.concatenate((history_y, y_test_block), axis=0)
            history_y = history_y[self.n_forecast_steps:]
            
            period += 1
        
        # Calculate overall metrics
        predictions = np.array(predictions)
        truth_final = truth[:len(predictions)] if len(truth) >= len(predictions) else truth
        
        overall_rmse = sqrt(mean_squared_error(truth_final, predictions))
        overall_mae = mean_absolute_error(truth_final, predictions)
        overall_r2 = r2_score(truth_final, predictions)
        
        # Store results
        self.results = {
            'model_type': self.model_type,
            'feature_type': self.feature_type,
            'predictions': predictions,
            'truth': truth_final,
            'period_errors': period_errors,
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'overall_r2': overall_r2,
            'n_periods': len(period_errors)
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"{'='*60}")
            print(f"Overall RMSE: {overall_rmse:.2f}")
            print(f"Overall MAE: {overall_mae:.2f}")
            print(f"Overall R²: {overall_r2:.4f}")
            print(f"Number of periods: {len(period_errors)}")
        
        # Save model if requested
        if save_model:
            if model_path is None:
                model_path = f"{self.model_type}_{self.feature_type}_model.h5"
            self.model.save(model_path)
            if self.verbose:
                print(f"Model saved to: {model_path}")
        
        return self.results
    
    def evaluate(self, data: pd.DataFrame, n_train: int = 35064) -> Dict:
        """
        Evaluate model on test data (alias for train, but can be extended).
        
        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed data with datetime index
        n_train : int
            Number of training samples
        
        Returns:
        --------
        Dict with evaluation results
        """
        return self.train(data, n_train=n_train, save_model=False)


def run_all_models(
    data_path: str = "data",
    years: Optional[List[int]] = None,
    download_from_drive: bool = False,
    drive_folder_id: Optional[str] = None,
    n_train: int = 35064,
    save_results: bool = True,
    results_path: str = "results"
) -> pd.DataFrame:
    """
    Run all model configurations and return comparison results.
    
    Parameters:
    -----------
    data_path : str
        Path to data directory
    years : List[int], optional
        Years to load
    download_from_drive : bool
        Whether to download from Google Drive
    drive_folder_id : str, optional
        Google Drive folder ID
    n_train : int
        Number of training samples
    save_results : bool
        Whether to save results to CSV
    results_path : str
        Directory to save results
    
    Returns:
    --------
    pd.DataFrame with comparison of all models
    """
    print("="*60)
    print("Loading and preprocessing data...")
    print("="*60)
    
    # Load and preprocess data
    if download_from_drive and drive_folder_id:
        data = preprocess_pipeline(
            data_path=data_path,
            download_from_drive=True,
            drive_folder_id=drive_folder_id,
            years=years
        )
    else:
        # Use existing preprocessing functions
        data = retrieve_data(data_path, years=years)
        data = handle_missing_values(data)
        data = data_and_aggregator(data)
        data = businesshour_and_we_generation(data)
        data = data.dropna()
    
    print(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Define model configurations
    configurations = [
        ('mlp', 'ts_only', 24),
        ('mlp', 'weekend', 1),
        ('mlp', 'business_hour', 1),
        ('lstm', 'ts_only', 1),
        ('lstm', 'weekend', 1),
        ('lstm', 'business_hour', 1),
        ('cnn', 'ts_only', 2),
        ('cnn', 'weekend', 2),
        ('cnn', 'business_hour', 2),
    ]
    
    all_results = []
    
    for model_type, feature_type, window_size in configurations:
        try:
            trainer = ModelTrainer(
                model_type=model_type,
                feature_type=feature_type,
                window_size=window_size,
                n_forecast_steps=24,
                epochs=100,
                batch_size=168,
                learning_rate=0.003,
                patience=3,
                verbose=True
            )
            
            results = trainer.train(data, n_train=n_train, save_model=False)
            
            all_results.append({
                'model': model_type.upper(),
                'features': feature_type,
                'window_size': window_size,
                'rmse': results['overall_rmse'],
                'mae': results['overall_mae'],
                'r2': results['overall_r2'],
                'n_periods': results['n_periods']
            })
            
        except Exception as e:
            print(f"Error training {model_type} with {feature_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('rmse')
    
    print("\n" + "="*60)
    print("ALL MODELS COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Save results
    if save_results:
        os.makedirs(results_path, exist_ok=True)
        results_file = os.path.join(results_path, "model_comparison.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
    
    return results_df


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train energy forecasting models')
    parser.add_argument('--data_path', type=str, default='data', help='Path to data directory')
    parser.add_argument('--download', action='store_true', help='Download from Google Drive')
    parser.add_argument('--folder_id', type=str, default=None, help='Google Drive folder ID')
    parser.add_argument('--years', type=int, nargs='+', default=None, help='Years to load')
    parser.add_argument('--n_train', type=int, default=35064, help='Number of training samples')
    parser.add_argument('--model', type=str, choices=['mlp', 'lstm', 'cnn', 'all'], default='all', help='Model to train')
    parser.add_argument('--features', type=str, choices=['ts_only', 'weekend', 'business_hour', 'all'], default='all', help='Features to use')
    
    args = parser.parse_args()
    
    if args.model == 'all' and args.features == 'all':
        # Run all configurations
        results = run_all_models(
            data_path=args.data_path,
            years=args.years,
            download_from_drive=args.download,
            drive_folder_id=args.folder_id or "1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H",
            n_train=args.n_train
        )
    else:
        # Run specific configuration
        print("Loading data...")
        data = preprocess_pipeline(
            data_path=args.data_path,
            download_from_drive=args.download,
            drive_folder_id=args.folder_id or "1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H",
            years=args.years
        )
        
        if args.features == 'all':
            feature_types = ['ts_only', 'weekend', 'business_hour']
        else:
            feature_types = [args.features]
        
        for feature_type in feature_types:
            trainer = ModelTrainer(
                model_type=args.model,
                feature_type=feature_type,
                window_size=24 if args.model == 'mlp' else (1 if args.model == 'lstm' else 2),
                verbose=True
            )
            trainer.train(data, n_train=args.n_train)

