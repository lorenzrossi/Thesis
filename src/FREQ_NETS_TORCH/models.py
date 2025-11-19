"""
Model architectures for energy time series forecasting using PyTorch.

This module defines the MLP, LSTM, and CNN model architectures used for forecasting.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MLP(nn.Module):
    """Multi-Layer Perceptron model for time series forecasting."""
    
    def __init__(
        self,
        input_size: int,
        hidden_units: int = 10,
        n_layers: int = 2,
        activation: str = 'relu',
        dropout_rate: Optional[float] = None,
        use_batch_norm: bool = False
    ):
        """
        Initialize MLP model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_units : int
            Number of units in hidden layers
        n_layers : int
            Number of hidden layers
        activation : str
            Activation function ('relu', 'tanh', 'sigmoid')
        dropout_rate : float, optional
            Dropout rate (None = no dropout)
        use_batch_norm : bool
            Whether to use batch normalization
        """
        super(MLP, self).__init__()
        
        layers = []
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        else:
            act_fn = nn.ReLU()
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_units))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_units))
        layers.append(act_fn)
        if dropout_rate:
            layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(act_fn)
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_units, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)


class LSTM(nn.Module):
    """Long Short-Term Memory model for time series forecasting."""
    
    def __init__(
        self,
        input_size: int,
        lstm_units: int = 10,
        n_layers: int = 2,
        dropout_rate: Optional[float] = None,
        return_sequences: bool = True
    ):
        """
        Initialize LSTM model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features per timestep
        lstm_units : int
            Number of LSTM units
        n_layers : int
            Number of LSTM layers
        dropout_rate : float, optional
            Dropout rate
        return_sequences : bool
            Whether to return sequences (True for stacked LSTMs)
        """
        super(LSTM, self).__init__()
        
        self.return_sequences = return_sequences
        self.n_layers = n_layers
        self.lstm_units = lstm_units
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate if dropout_rate else 0.0
        )
        
        # Output layer
        # For return_sequences=True, we flatten all timesteps and then apply linear layer
        # This matches TensorFlow's Flatten() + Dense(1) behavior
        self.return_sequences = return_sequences
        if return_sequences:
            # In practice, LSTM uses window_size=1, so seq_len=1
            # Flattened size = seq_len * lstm_units = 1 * lstm_units = lstm_units
            # But to be safe for other window sizes, we use a two-step approach:
            # 1. Apply linear to each timestep: (batch, seq_len, lstm_units) -> (batch, seq_len, 1)
            # 2. Flatten and take mean (or use adaptive pooling)
            # Actually, to match TensorFlow exactly: Flatten() + Dense(1)
            # We'll use a workaround: apply linear to each timestep, then flatten and use another linear
            # But that requires knowing seq_len. Instead, let's use adaptive pooling
            self.fc_per_timestep = nn.Linear(lstm_units, 1)
            # After applying fc_per_timestep, we get (batch, seq_len, 1)
            # Then we flatten to (batch, seq_len) and need to reduce to (batch, 1)
            # For TensorFlow compatibility, we'll use mean (which is equivalent for seq_len=1)
            self.use_mean = True
        else:
            self.fc = nn.Linear(lstm_units, 1)
    
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, lstm_units)
        
        if self.return_sequences:
            # Match TensorFlow: Flatten() + Dense(1)
            # Step 1: Apply linear to each timestep
            output = self.fc_per_timestep(lstm_out)  # (batch, seq_len, 1)
            # Step 2: Flatten: (batch, seq_len, 1) -> (batch, seq_len)
            flattened = output.squeeze(-1)  # (batch, seq_len)
            # Step 3: Reduce to single output per sample
            # For TensorFlow compatibility with Flatten() + Dense(1), we take mean
            # (In practice with window_size=1, this is just the single value)
            return flattened.mean(dim=1, keepdim=True)  # (batch, 1)
        else:
            # Use only last timestep
            return self.fc(lstm_out[:, -1, :])


class CNN(nn.Module):
    """1D Convolutional Neural Network model for time series forecasting."""
    
    def __init__(
        self,
        input_size: int,
        filters: int = 16,
        kernel_size: int = 1,
        n_layers: int = 2,
        activation: str = 'relu',
        dropout_rate: Optional[float] = None,
        use_pooling: bool = False,
        pool_size: int = 2
    ):
        """
        Initialize CNN model.
        
        Parameters:
        -----------
        input_size : int
            Number of input features per timestep
        filters : int
            Number of convolutional filters
        kernel_size : int
            Size of convolutional kernel
        n_layers : int
            Number of convolutional layers
        activation : str
            Activation function
        dropout_rate : float, optional
            Dropout rate
        use_pooling : bool
            Whether to use max pooling
        pool_size : int
            Pool size for max pooling
        """
        super(CNN, self).__init__()
        
        layers = []
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            act_fn = nn.ReLU()
        
        # First convolutional layer
        layers.append(nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        ))
        layers.append(act_fn)
        if dropout_rate:
            layers.append(nn.Dropout(dropout_rate))
        if use_pooling:
            layers.append(nn.MaxPool1d(pool_size))
        
        # Additional convolutional layers
        for _ in range(n_layers - 1):
            layers.append(nn.Conv1d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            layers.append(act_fn)
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            if use_pooling:
                layers.append(nn.MaxPool1d(pool_size))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Flatten and output layer
        # We'll determine the flattened size dynamically or use adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(filters, 1)
    
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch, seq_len, features) -> need (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Adaptive pooling to get fixed size
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)  # Remove sequence dimension
        
        # Output layer
        return self.fc(x)


class ModelBuilder:
    """Builder class for creating different neural network architectures."""
    
    @staticmethod
    def build_mlp(
        input_size: int,
        hidden_units: int = 10,
        n_layers: int = 2,
        activation: str = 'relu',
        dropout_rate: Optional[float] = None,
        use_batch_norm: bool = False
    ) -> MLP:
        """Build a Multi-Layer Perceptron (MLP) model."""
        return MLP(
            input_size=input_size,
            hidden_units=hidden_units,
            n_layers=n_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
    
    @staticmethod
    def build_lstm(
        input_size: int,
        lstm_units: int = 10,
        n_layers: int = 2,
        dropout_rate: Optional[float] = None,
        return_sequences: bool = True
    ) -> LSTM:
        """Build a Long Short-Term Memory (LSTM) model."""
        return LSTM(
            input_size=input_size,
            lstm_units=lstm_units,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            return_sequences=return_sequences
        )
    
    @staticmethod
    def build_cnn(
        input_size: int,
        filters: int = 16,
        kernel_size: int = 1,
        n_layers: int = 2,
        activation: str = 'relu',
        dropout_rate: Optional[float] = None,
        use_pooling: bool = False,
        pool_size: int = 2
    ) -> CNN:
        """Build a 1D Convolutional Neural Network (CNN) model."""
        return CNN(
            input_size=input_size,
            filters=filters,
            kernel_size=kernel_size,
            n_layers=n_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            use_pooling=use_pooling,
            pool_size=pool_size
        )


def create_model(
    model_type: str,
    input_size: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a model based on type.
    
    Parameters:
    -----------
    model_type : str
        Type of model: 'mlp', 'lstm', or 'cnn'
    input_size : int
        Input size (number of features)
    **kwargs
        Additional arguments passed to model builder
    
    Returns:
    --------
    PyTorch nn.Module
    """
    builder = ModelBuilder()
    
    if model_type.lower() == 'mlp':
        return builder.build_mlp(input_size, **kwargs)
    elif model_type.lower() == 'lstm':
        return builder.build_lstm(input_size, **kwargs)
    elif model_type.lower() == 'cnn':
        return builder.build_cnn(input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'mlp', 'lstm', 'cnn'")


# Default model configurations matching the TensorFlow versions
DEFAULT_CONFIGS = {
    'mlp': {
        'hidden_units': 10,
        'n_layers': 2,
        'activation': 'relu',
        'dropout_rate': None,
        'use_batch_norm': False
    },
    'lstm': {
        'lstm_units': 10,
        'n_layers': 2,
        'dropout_rate': None,
        'return_sequences': True
    },
    'cnn': {
        'filters': 16,
        'kernel_size': 1,
        'n_layers': 2,
        'activation': 'relu',
        'dropout_rate': None,
        'use_pooling': False
    }
}

