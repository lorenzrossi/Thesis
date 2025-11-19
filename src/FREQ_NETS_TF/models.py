"""
Model architectures for energy time series forecasting.

This module defines the MLP, LSTM, and CNN model architectures used for forecasting.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional


class ModelBuilder:
    """Builder class for creating different neural network architectures."""
    
    @staticmethod
    def build_mlp(
        input_shape: Tuple[int, ...],
        hidden_units: int = 10,
        n_layers: int = 2,
        activation: str = 'relu',
        dropout_rate: Optional[float] = None,
        use_batch_norm: bool = False
    ) -> Sequential:
        """
        Build a Multi-Layer Perceptron (MLP) model.
        
        Parameters:
        -----------
        input_shape : Tuple
            Shape of input data (n_features,)
        hidden_units : int
            Number of units in hidden layers
        n_layers : int
            Number of hidden layers
        activation : str
            Activation function
        dropout_rate : float, optional
            Dropout rate (None = no dropout)
        use_batch_norm : bool
            Whether to use batch normalization
        
        Returns:
        --------
        Sequential model
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_units, activation=activation, input_shape=input_shape))
        if use_batch_norm:
            model.add(BatchNormalization())
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            model.add(Dense(hidden_units, activation=activation))
            if use_batch_norm:
                model.add(BatchNormalization())
            if dropout_rate:
                model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        return model
    
    @staticmethod
    def build_lstm(
        input_shape: Tuple[int, int],
        lstm_units: int = 10,
        n_layers: int = 2,
        dropout_rate: Optional[float] = None,
        recurrent_dropout: Optional[float] = None,
        return_sequences: bool = True
    ) -> Sequential:
        """
        Build a Long Short-Term Memory (LSTM) model.
        
        Parameters:
        -----------
        input_shape : Tuple[int, int]
            Shape of input data (timesteps, n_features)
        lstm_units : int
            Number of LSTM units
        n_layers : int
            Number of LSTM layers
        dropout_rate : float, optional
            Dropout rate for inputs
        recurrent_dropout : float, optional
            Dropout rate for recurrent connections
        return_sequences : bool
            Whether to return sequences (True for stacked LSTMs)
        
        Returns:
        --------
        Sequential model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            lstm_units,
            return_sequences=(n_layers > 1 or return_sequences),
            input_shape=input_shape,
            dropout=dropout_rate if dropout_rate else 0.0,
            recurrent_dropout=recurrent_dropout if recurrent_dropout else 0.0
        ))
        
        # Additional LSTM layers
        for i in range(1, n_layers):
            is_last = (i == n_layers - 1)
            model.add(LSTM(
                lstm_units,
                return_sequences=(not is_last and return_sequences),
                dropout=dropout_rate if dropout_rate else 0.0,
                recurrent_dropout=recurrent_dropout if recurrent_dropout else 0.0
            ))
        
        # Flatten if needed
        if return_sequences:
            model.add(Flatten())
        
        # Output layer
        model.add(Dense(1))
        
        return model
    
    @staticmethod
    def build_cnn(
        input_shape: Tuple[int, int],
        filters: int = 16,
        kernel_size: int = 1,
        n_layers: int = 2,
        activation: str = 'relu',
        dropout_rate: Optional[float] = None,
        use_pooling: bool = False,
        pool_size: int = 2
    ) -> Sequential:
        """
        Build a 1D Convolutional Neural Network (CNN) model.
        
        Parameters:
        -----------
        input_shape : Tuple[int, int]
            Shape of input data (timesteps, n_features)
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
        
        Returns:
        --------
        Sequential model
        """
        model = Sequential()
        
        # First convolutional layer
        model.add(Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            input_shape=input_shape
        ))
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        if use_pooling:
            from tensorflow.keras.layers import MaxPooling1D
            model.add(MaxPooling1D(pool_size=pool_size))
        
        # Additional convolutional layers
        for _ in range(n_layers - 1):
            model.add(Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation=activation
            ))
            if dropout_rate:
                model.add(Dropout(dropout_rate))
            if use_pooling:
                from tensorflow.keras.layers import MaxPooling1D
                model.add(MaxPooling1D(pool_size=pool_size))
        
        # Flatten
        model.add(Flatten())
        
        # Output layer
        model.add(Dense(1))
        
        return model
    
    @staticmethod
    def compile_model(
        model: Sequential,
        learning_rate: float = 0.003,
        optimizer: Optional[Adam] = None,
        loss: str = 'mse',
        metrics: Optional[list] = None
    ) -> Sequential:
        """
        Compile a Keras model with standard settings.
        
        Parameters:
        -----------
        model : Sequential
            Model to compile
        learning_rate : float
            Learning rate for optimizer
        optimizer : Adam, optional
            Custom optimizer (if None, creates new Adam optimizer)
        loss : str
            Loss function name
        metrics : list, optional
            List of metrics to track
        
        Returns:
        --------
        Compiled Sequential model
        """
        if optimizer is None:
            optimizer = Adam(learning_rate=learning_rate)
        
        if metrics is None:
            metrics = [tf.keras.metrics.RootMeanSquaredError()]
        
        if loss == 'mse':
            loss = tf.keras.losses.MeanSquaredError()
        elif loss == 'mae':
            loss = tf.keras.losses.MeanAbsoluteError()
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model


def create_model(
    model_type: str,
    input_shape: Tuple,
    **kwargs
) -> Sequential:
    """
    Factory function to create a model based on type.
    
    Parameters:
    -----------
    model_type : str
        Type of model: 'mlp', 'lstm', or 'cnn'
    input_shape : Tuple
        Input shape for the model
    **kwargs
        Additional arguments passed to model builder
    
    Returns:
    --------
    Compiled Sequential model
    """
    builder = ModelBuilder()
    
    if model_type.lower() == 'mlp':
        model = builder.build_mlp(input_shape, **kwargs)
    elif model_type.lower() == 'lstm':
        model = builder.build_lstm(input_shape, **kwargs)
    elif model_type.lower() == 'cnn':
        model = builder.build_cnn(input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'mlp', 'lstm', 'cnn'")
    
    # Compile model with default settings (can be overridden later)
    learning_rate = kwargs.get('learning_rate', 0.003)
    model = builder.compile_model(model, learning_rate=learning_rate)
    
    return model


# Default model configurations matching the original notebooks
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
        'recurrent_dropout': None,
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

