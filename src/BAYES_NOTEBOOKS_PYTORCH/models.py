"""
Bayesian neural network architectures for energy time series forecasting using PyTorch.

This module defines Bayesian MLP models using the blitz library for uncertainty quantification.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from blitz.modules import BayesianLinear
    from blitz.utils import variational_estimator
    BLITZ_AVAILABLE = True
except ImportError:
    BLITZ_AVAILABLE = False
    print("Warning: blitz-bayesian-pytorch not installed. Install with: pip install blitz-bayesian-pytorch")


@variational_estimator
class BayesianMLP(nn.Module):
    """
    Bayesian Multi-Layer Perceptron for time series forecasting.
    
    Uses variational inference to learn weight distributions, providing
    uncertainty quantification in predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_units: int = 10,
        n_layers: int = 2,
        prior_sigma_1: float = 0.01,
        prior_sigma_2: float = 0.01,
        prior_pi: float = 1.0,
        posterior_mu_init: float = 0.0
    ):
        """
        Initialize Bayesian MLP model.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        output_dim : int
            Number of output features (default: 1 for regression)
        hidden_units : int
            Number of units in hidden layers (default: 10)
        n_layers : int
            Number of hidden layers (default: 2)
        prior_sigma_1 : float
            Prior standard deviation 1 (default: 0.01)
        prior_sigma_2 : float
            Prior standard deviation 2 (default: 0.01)
        prior_pi : float
            Prior mixture weight (default: 1.0)
        posterior_mu_init : float
            Initial mean for posterior distribution (default: 0.0)
        """
        if not BLITZ_AVAILABLE:
            raise ImportError("blitz-bayesian-pytorch is required. Install with: pip install blitz-bayesian-pytorch")
        
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        
        layers = []
        
        # Input layer
        layers.append(
            BayesianLinear(
                input_dim, 
                hidden_units,
                prior_sigma_1=prior_sigma_1,
                prior_sigma_2=prior_sigma_2,
                prior_pi=prior_pi,
                posterior_mu_init=posterior_mu_init
            )
        )
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(BayesianLinear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(BayesianLinear(hidden_units, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the Bayesian network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        return self.layers(x)


class ModelBuilder:
    """Builder class for creating Bayesian neural network architectures."""
    
    @staticmethod
    def build_bayesian_mlp(
        input_dim: int,
        output_dim: int = 1,
        hidden_units: int = 10,
        n_layers: int = 2,
        prior_sigma_1: float = 0.01,
        prior_sigma_2: float = 0.01,
        prior_pi: float = 1.0,
        posterior_mu_init: float = 0.0
    ) -> BayesianMLP:
        """
        Build a Bayesian Multi-Layer Perceptron (BMLP) model.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        output_dim : int
            Number of output features
        hidden_units : int
            Number of units in hidden layers
        n_layers : int
            Number of hidden layers
        prior_sigma_1 : float
            Prior standard deviation 1
        prior_sigma_2 : float
            Prior standard deviation 2
        prior_pi : float
            Prior mixture weight
        posterior_mu_init : float
            Initial mean for posterior distribution
        
        Returns:
        --------
        BayesianMLP
            Bayesian MLP model
        """
        return BayesianMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            n_layers=n_layers,
            prior_sigma_1=prior_sigma_1,
            prior_sigma_2=prior_sigma_2,
            prior_pi=prior_pi,
            posterior_mu_init=posterior_mu_init
        )


def create_model(
    model_type: str,
    input_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a Bayesian model based on type.
    
    Parameters:
    -----------
    model_type : str
        Type of model: 'bayesian_mlp' or 'bmlp'
    input_dim : int
        Input dimension (number of features)
    **kwargs
        Additional arguments passed to model builder
    
    Returns:
    --------
    PyTorch nn.Module (Bayesian model)
    """
    builder = ModelBuilder()
    
    if model_type.lower() in ['bayesian_mlp', 'bmlp']:
        return builder.build_bayesian_mlp(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'bayesian_mlp' or 'bmlp'")


# Default model configurations matching the original notebooks
DEFAULT_CONFIGS = {
    'bayesian_mlp': {
        'output_dim': 1,
        'hidden_units': 10,
        'n_layers': 2,
        'prior_sigma_1': 0.01,
        'prior_sigma_2': 0.01,
        'prior_pi': 1.0,
        'posterior_mu_init': 0.0
    },
    'bmlp': {  # Alias
        'output_dim': 1,
        'hidden_units': 10,
        'n_layers': 2,
        'prior_sigma_1': 0.01,
        'prior_sigma_2': 0.01,
        'prior_pi': 1.0,
        'posterior_mu_init': 0.0
    }
}

