"""
Training and testing module for Bayesian energy forecasting models using PyTorch.

This module handles data preparation, rolling window training, and evaluation with
uncertainty quantification for Bayesian neural networks.
"""

import os
import sys
import numpy as np
import pandas as pd
from math import sqrt
from typing import Dict, List, Tuple, Optional, Literal
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
    """Handles data preparation for Bayesian MLP models."""
    
    def __init__(
        self,
        feature_type: Literal['ts_only', 'weekend', 'business_hour'],
        window_size: int = 1  # Number of lags for MLP (typically 1 for lag_1)
    ):
        """
        Initialize data preparator.
        
        Parameters:
        -----------
        feature_type : str
            Feature combination: 'ts_only', 'weekend', or 'business_hour'
        window_size : int
            Number of lags to use (default: 1 for lag_1)
        """
        self.feature_type = feature_type
        self.window_size = window_size
        self.scaler = None
        self.target_scaler = None
        self.feature_names = []
    
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
    
    def scale_target(self, target: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale target variable separately.
        """
        if fit:
            self.target_scaler = StandardScaler()
            self.target_scaler.fit(target.reshape(-1, 1))
        
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def prepare_data_for_model(
        self, 
        train_data: np.ndarray, 
        test_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for Bayesian MLP model.
        Creates lag features and combines with dummy variables.
        
        Returns:
        --------
        X_train, y_train, X_test, y_test
        """
        # Create DataFrames
        df_train = pd.DataFrame(train_data, columns=[f'feat_{i}' for i in range(train_data.shape[1])])
        df_test = pd.DataFrame(test_data, columns=[f'feat_{i}' for i in range(test_data.shape[1])])
        
        # Create lags for the first column (total_aggregated)
        # For Bayesian models, typically use lag_1 only
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
        
        return X_train, y_train, X_test, y_test


class BayesianModelTrainer:
    """Handles training and evaluation of Bayesian forecasting models with rolling window approach."""
    
    def __init__(
        self,
        feature_type: Literal['ts_only', 'weekend', 'business_hour'],
        window_size: int = 1,  # Number of lags (typically 1 for lag_1)
        n_forecast_steps: int = 24,
        epochs: int = 50,  # Increased default for proper training (was 5, but 50+ recommended)
        batch_size: int = 168,
        learning_rate: float = 0.01,
        verbose: bool = True,
        device: Optional[str] = None,
        sample_nbr: int = 5,  # Number of samples for ELBO during training
        inference_samples: int = 100,  # Number of samples for prediction
        std_multiplier: float = 5.0,  # Multiplier for confidence intervals
        complexity_cost_weight: Optional[float] = None,  # If None, will be set to 1/n_train
        **model_kwargs
    ):
        """
        Initialize Bayesian model trainer.
        
        Parameters:
        -----------
        feature_type : str
            Feature combination: 'ts_only', 'weekend', or 'business_hour'
        window_size : int
            Number of lags to use (default: 1 for lag_1)
        n_forecast_steps : int
            Number of steps to forecast ahead
        epochs : int
            Number of training epochs per window
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        verbose : bool
            Whether to print progress
        device : str, optional
            Device to use ('cpu', 'cuda', 'mps', or None for auto-detect)
        sample_nbr : int
            Number of samples for ELBO calculation during training
        inference_samples : int
            Number of Monte Carlo samples for prediction
        std_multiplier : float
            Multiplier for confidence intervals (default: 5.0)
        complexity_cost_weight : float, optional
            Weight for KL divergence term. If None, uses 1/n_train
        **model_kwargs
            Additional arguments for model creation
        """
        self.feature_type = feature_type
        self.window_size = window_size
        self.n_forecast_steps = n_forecast_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.sample_nbr = sample_nbr
        self.inference_samples = inference_samples
        self.std_multiplier = std_multiplier
        self.complexity_cost_weight = complexity_cost_weight
        self.model_kwargs = model_kwargs
        
        # Device setup (supports CUDA, MPS for Apple Silicon, or CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')  # Apple Silicon GPU
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.data_preparator = DataPreparator(feature_type, window_size)
        self.results = {}
    
    def _train_epoch(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        n_train: int
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch using ELBO loss.
        
        Returns:
        --------
        elbo, mse, kl_div : Tuple of average losses
        """
        model.train()
        elbo_total = 0.0
        mse_total = 0.0
        kl_div_total = 0.0
        n_batches = 0
        
        # Set complexity cost weight if not provided
        complexity_weight = self.complexity_cost_weight
        if complexity_weight is None:
            complexity_weight = 1.0 / n_train
        
        for datapoints, labels in dataloader:
            datapoints = datapoints.to(self.device).float()
            labels = labels.to(self.device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Calculate ELBO loss (returns [something, elbo, mse, kl_div] based on notebooks)
            loss = model.sample_elbo_detailed_loss(
                inputs=datapoints,
                labels=labels,
                criterion=criterion,
                sample_nbr=self.sample_nbr,
                complexity_cost_weight=complexity_weight
            )
            
            # Based on notebooks: loss[1] is ELBO, loss[2] is MSE, loss[3] is KL
            # Ensure loss components are tensors (not numpy arrays)
            if isinstance(loss, (list, tuple)):
                elbo_loss = loss[1] if len(loss) > 1 else loss[0]
                mse_loss = loss[2] if len(loss) > 2 else torch.tensor(0.0)
                kl_loss = loss[3] if len(loss) > 3 else torch.tensor(0.0)
            else:
                elbo_loss = loss
                mse_loss = torch.tensor(0.0)
                kl_loss = torch.tensor(0.0)
            
            # Ensure tensors are on correct device
            if not isinstance(elbo_loss, torch.Tensor):
                elbo_loss = torch.tensor(elbo_loss, device=self.device, requires_grad=True)
            if not isinstance(mse_loss, torch.Tensor):
                mse_loss = torch.tensor(mse_loss, device=self.device)
            if not isinstance(kl_loss, torch.Tensor):
                kl_loss = torch.tensor(kl_loss, device=self.device)
            
            # Backward pass on ELBO
            elbo_loss.backward()
            optimizer.step()
            
            elbo_total += elbo_loss.item() if isinstance(elbo_loss, torch.Tensor) else float(elbo_loss)
            mse_total += mse_loss.item() if isinstance(mse_loss, torch.Tensor) else float(mse_loss)
            kl_div_total += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else float(kl_loss)
            n_batches += 1
        
        return (
            elbo_total / n_batches if n_batches > 0 else 0.0,
            mse_total / n_batches if n_batches > 0 else 0.0,
            kl_div_total / n_batches if n_batches > 0 else 0.0
        )
    
    def _evaluate_regression(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate Bayesian model with uncertainty quantification.
        
        Parameters:
        -----------
        model : nn.Module
            Bayesian model
        X : np.ndarray
            Input features
        y : np.ndarray, optional
            True values (scaled, for comparison)
        
        Returns:
        --------
        y_pred, y_true, ci_upper, ci_lower, stds (all in original scale)
        """
        model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Monte Carlo sampling for uncertainty quantification
        with torch.no_grad():
            preds = [model(X_tensor) for _ in range(self.inference_samples)]
            preds = torch.stack(preds)
        
        # Calculate statistics in scaled space
        means_scaled = preds.mean(axis=0).cpu().numpy()
        stds_scaled = preds.std(axis=0).cpu().numpy()
        
        # Confidence intervals in scaled space
        ci_upper_scaled = means_scaled + (self.std_multiplier * stds_scaled)
        ci_lower_scaled = means_scaled - (self.std_multiplier * stds_scaled)
        
        # Inverse transform to original scale
        y_pred = self.data_preparator.target_scaler.inverse_transform(means_scaled).flatten()
        ci_upper = self.data_preparator.target_scaler.inverse_transform(ci_upper_scaled).flatten()
        ci_lower = self.data_preparator.target_scaler.inverse_transform(ci_lower_scaled).flatten()
        
        # Inverse transform true values if provided (y is already scaled)
        y_true = None
        if y is not None:
            y_true = self.data_preparator.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        
        # Note: stds are in scaled space, convert to original scale approximation
        # This is an approximation - for exact std in original scale, would need to transform each sample
        stds_original = stds_scaled.flatten() * self.data_preparator.target_scaler.scale_[0]
        
        return y_pred, y_true, ci_upper, ci_lower, stds_original
    
    def train(
        self, 
        data: pd.DataFrame,
        n_train: int = 35064,  # 3 years of hourly data
        save_model: bool = False,
        model_path: Optional[str] = None
    ) -> Dict:
        """
        Train Bayesian model using rolling window approach.
        
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
        Dict with training results, metrics, and uncertainty information
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training Bayesian MLP model with {self.feature_type} features")
            print(f"Device: {self.device}")
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
        
        # Scale target separately
        y_train_scaled = self.data_preparator.scale_target(y_train, fit=True)
        y_test_scaled = self.data_preparator.scale_target(y_test, fit=False)
        
        # Get truth values (unscaled) for evaluation
        truth_start_idx = self.window_size
        truth = test_data_raw[truth_start_idx:truth_start_idx + len(y_test), 0] if len(y_test) > 0 else test_data_raw[truth_start_idx:, 0]
        
        # Ensure truth has the same length as y_test
        if len(truth) > len(y_test):
            truth = truth[:len(y_test)]
        elif len(truth) < len(y_test):
            truth = np.pad(truth, (0, len(y_test) - len(truth)), mode='edge')
        
        # Determine input size
        input_dim = X_train.shape[1]
        
        # Merge default config with user kwargs
        config = DEFAULT_CONFIGS['bayesian_mlp'].copy()
        config.update(self.model_kwargs)
        
        # Create model
        self.model = create_model('bayesian_mlp', input_dim, **config)
        self.model = self.model.to(self.device)
        
        if self.verbose:
            print(f"\nModel Architecture:")
            print(self.model)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Rolling window training
        history_X = torch.FloatTensor(X_train).to(self.device)
        history_y = torch.FloatTensor(y_train_scaled).to(self.device)
        
        predictions = []
        ci_uppers = []
        ci_lowers = []
        stds_list = []
        period_errors = []
        period = 1
        
        # Adjust test data iteration based on available data
        test_iterations = range(0, len(X_test), self.n_forecast_steps)
        
        if len(list(test_iterations)) == 0:
            warnings.warn("No test data available for rolling window training. Using train data for evaluation.")
            test_iterations = [0]
            X_test = X_train[-self.n_forecast_steps:] if len(X_train) >= self.n_forecast_steps else X_train
            y_test_scaled = y_train_scaled[-self.n_forecast_steps:] if len(y_train_scaled) >= self.n_forecast_steps else y_train_scaled
            truth = train_data_raw[-len(y_test_scaled):, 0] if len(y_test_scaled) > 0 else train_data_raw[:, 0]
        
        for i in test_iterations:
            t_end = min(i + self.n_forecast_steps, len(X_test))
            X_test_block = X_test[i:t_end]
            y_test_block = y_test_scaled[i:t_end]
            truth_block = truth[i:t_end] if len(truth) > i else truth[-len(y_test_block):]
            
            if self.verbose:
                print(f"\nTraining and predicting for period {period} (samples {i} to {t_end})")
            
            # Create DataLoader for training
            dataset = TensorDataset(history_X, history_y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Training loop
            for epoch in range(self.epochs):
                elbo, mse, kl_div = self._train_epoch(
                    self.model, dataloader, criterion, optimizer, len(history_X)
                )
                
                if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - ELBO: {elbo:.6f}, MSE: {mse:.6f}, KL Div: {kl_div:.6f}")
            
            # Make predictions with uncertainty
            y_pred, y_true_block, ci_upper, ci_lower, stds = self._evaluate_regression(
                self.model, X_test_block, y_test_block
            )
            
            # Calculate period RMSE
            period_rmse = sqrt(np.mean((truth_block - y_pred)**2))
            period_errors.append(period_rmse)
            predictions.extend(y_pred)
            ci_uppers.extend(ci_upper)
            ci_lowers.extend(ci_lower)
            stds_list.extend(stds)
            
            if self.verbose:
                print(f"Period RMSE: {period_rmse:.2f}")
            
            # Update history with actual observed values
            X_test_tensor = torch.FloatTensor(X_test_block).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test_block).to(self.device)
            
            history_X = torch.cat([history_X, X_test_tensor], dim=0)
            history_X = history_X[self.n_forecast_steps:]
            history_y = torch.cat([history_y, y_test_tensor], dim=0)
            history_y = history_y[self.n_forecast_steps:]
            
            period += 1
        
        # Calculate overall metrics
        predictions = np.array(predictions)
        truth_final = truth[:len(predictions)] if len(truth) >= len(predictions) else truth
        
        overall_rmse = sqrt(mean_squared_error(truth_final, predictions))
        overall_mae = mean_absolute_error(truth_final, predictions)
        overall_r2 = r2_score(truth_final, predictions)
        
        # Calculate coverage (percentage of true values within confidence intervals)
        ci_upper_array = np.array(ci_uppers)[:len(truth_final)]
        ci_lower_array = np.array(ci_lowers)[:len(truth_final)]
        coverage = np.mean((ci_lower_array <= truth_final) & (truth_final <= ci_upper_array))
        
        # Store results
        self.results = {
            'model_type': 'bayesian_mlp',
            'feature_type': self.feature_type,
            'predictions': predictions,
            'truth': truth_final,
            'ci_upper': ci_upper_array,
            'ci_lower': ci_lower_array,
            'stds': np.array(stds_list)[:len(truth_final)],
            'period_errors': period_errors,
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'overall_r2': overall_r2,
            'coverage': coverage,  # Percentage of true values within CI
            'n_periods': len(period_errors)
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"{'='*60}")
            print(f"Overall RMSE: {overall_rmse:.2f}")
            print(f"Overall MAE: {overall_mae:.2f}")
            print(f"Overall R²: {overall_r2:.4f}")
            print(f"Coverage: {coverage:.2%} (true values within {self.std_multiplier}σ CI)")
            print(f"Number of periods: {len(period_errors)}")
        
        # Save model if requested
        if save_model:
            if model_path is None:
                model_path = f"bayesian_mlp_{self.feature_type}_model.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': 'bayesian_mlp',
                'input_dim': input_dim,
                'config': config,
                'feature_type': self.feature_type
            }, model_path)
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
    results_path: str = "results",
    device: Optional[str] = None
) -> pd.DataFrame:
    """
    Run all Bayesian model configurations and return comparison results.
    
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
    device : str, optional
        Device to use ('cpu', 'cuda', 'mps', or None for auto-detect)
    
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
        ('ts_only', 1),
        ('weekend', 1),
        ('business_hour', 1),
    ]
    
    all_results = []
    
    for feature_type, window_size in configurations:
        try:
            trainer = BayesianModelTrainer(
                feature_type=feature_type,
                window_size=window_size,
                n_forecast_steps=24,
                epochs=5,
                batch_size=168,
                learning_rate=0.01,
                verbose=True,
                device=device
            )
            
            results = trainer.train(data, n_train=n_train, save_model=False)
            
            all_results.append({
                'features': feature_type,
                'window_size': window_size,
                'rmse': results['overall_rmse'],
                'mae': results['overall_mae'],
                'r2': results['overall_r2'],
                'coverage': results['coverage'],
                'n_periods': results['n_periods']
            })
            
        except Exception as e:
            print(f"Error training Bayesian MLP with {feature_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('rmse')
    
    print("\n" + "="*60)
    print("ALL BAYESIAN MODELS COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Save results
    if save_results:
        os.makedirs(results_path, exist_ok=True)
        results_file = os.path.join(results_path, "bayesian_model_comparison.csv")
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
    
    return results_df


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Bayesian energy forecasting models with PyTorch')
    parser.add_argument('--data_path', type=str, default='data', help='Path to data directory')
    parser.add_argument('--download', action='store_true', help='Download from Google Drive')
    parser.add_argument('--folder_id', type=str, default=None, help='Google Drive folder ID')
    parser.add_argument('--years', type=int, nargs='+', default=None, help='Years to load')
    parser.add_argument('--n_train', type=int, default=35064, help='Number of training samples')
    parser.add_argument('--features', type=str, choices=['ts_only', 'weekend', 'business_hour', 'all'], default='all', help='Features to use')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, mps, or None for auto)')
    
    args = parser.parse_args()
    
    if args.features == 'all':
        # Run all configurations
        results = run_all_models(
            data_path=args.data_path,
            years=args.years,
            download_from_drive=args.download,
            drive_folder_id=args.folder_id or "1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H",
            n_train=args.n_train,
            device=args.device
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
        
        trainer = BayesianModelTrainer(
            feature_type=args.features,
            window_size=1,
            verbose=True,
            device=args.device
        )
        trainer.train(data, n_train=args.n_train)

