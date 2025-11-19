"""
Manual implementation of Bayesian Neural Networks using BayesByBackprop methodology.

This module implements Bayesian neural networks from scratch using the BayesByBackprop
algorithm introduced by Blundell et al. (2015) in "Weight Uncertainty in Neural Networks".

This implementation follows the style of the BayesianDeepWine notebook, using
torch.distributions for cleaner and more PyTorch-native code.

Reference:
    Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015).
    Weight uncertainty in neural networks. ICML 2015.
    https://arxiv.org/abs/1505.05424

Key Features:
    - Manual implementation of BayesianLinear layer using torch.distributions
    - Reparameterization trick via q.rsample() (automatic differentiation)
    - KL divergence calculation using torch.distributions.kl.kl_divergence
    - ELBO (Evidence Lower BOund) loss function
    - Standard parameters from literature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from typing import Optional, Tuple
import math


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer implementing BayesByBackprop.
    
    Each weight and bias is treated as a probability distribution (Gaussian)
    rather than a fixed value. During forward pass, weights are sampled from
    the learned posterior distribution using the reparameterization trick.
    
    Parameters:
    -----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    prior_mu : float
        Mean of the prior distribution (default: 0.0, standard in literature)
    prior_sigma : float
        Standard deviation of the prior distribution (default: 1.0, standard)
    posterior_mu_init : float
        Initial mean for posterior distribution (default: 0.0)
    posterior_rho_init : float
        Initial value for rho (log variance parameter) (default: -3.0)
        This gives initial sigma ≈ 0.05 after softplus transformation
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0
    ):
        super(BayesianLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Prior distribution parameters (fixed, not learned)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        
        # Posterior distribution parameters (learned)
        # We learn mu (mean) and rho (log variance) for each weight
        # Using rho instead of log(sigma^2) directly for numerical stability
        
        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).normal_(posterior_mu_init, 0.1)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features).normal_(posterior_rho_init, 0.1)
        )
        
        # Bias parameters
        self.bias_mu = nn.Parameter(
            torch.empty(out_features).normal_(posterior_mu_init, 0.1)
        )
        self.bias_rho = nn.Parameter(
            torch.empty(out_features).normal_(posterior_rho_init, 0.1)
        )
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters with small random values."""
        # Weights already initialized in __init__, but can add custom initialization here
        pass
    
    def _get_weight_sigma(self) -> torch.Tensor:
        """
        Convert rho to sigma using softplus transformation.
        
        sigma = log(1 + exp(rho))
        
        This ensures sigma > 0 and is numerically stable.
        Same as torch.log(1. + torch.exp(self.weight_rho))
        """
        return torch.log(1 + torch.exp(self.weight_rho))
    
    def _get_bias_sigma(self) -> torch.Tensor:
        """Convert rho to sigma for bias."""
        return torch.log(1 + torch.exp(self.bias_rho))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using reparameterization trick via torch.distributions.
        
        Uses q.rsample() which automatically handles the reparameterization trick:
        w = mu + sigma * epsilon, where epsilon ~ N(0, 1)
        
        This allows gradients to flow through the sampling process.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features)
        
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, out_features)
        """
        # Create posterior distribution q(w|θ) = N(μ, σ²)
        weight_sigma = self._get_weight_sigma()
        bias_sigma = self._get_bias_sigma()
        
        # Use torch.distributions for cleaner code (like BayesianDeepWine notebook)
        weight_q = D.Normal(loc=self.weight_mu, scale=weight_sigma)
        bias_q = D.Normal(loc=self.bias_mu, scale=bias_sigma)
        
        # Sample using reparameterization trick (rsample = reparameterized sample)
        # This is differentiable and allows gradients to flow
        weight = weight_q.rsample()
        bias = bias_q.rsample()
        
        # Standard linear transformation
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior q(w|θ) and prior p(w).
        
        Uses torch.distributions.kl.kl_divergence for cleaner implementation,
        matching the style of the BayesianDeepWine notebook.
        
        KL(q(w|θ) || p(w)) where:
        - q(w|θ) = N(μ, σ²) is the posterior (learned distribution)
        - p(w) = N(μₚ, σₚ²) is the prior (fixed distribution)
        
        Returns:
        --------
        torch.Tensor
            Total KL divergence (scalar)
        """
        # Create posterior distributions
        weight_sigma = self._get_weight_sigma()
        bias_sigma = self._get_bias_sigma()
        
        weight_q = D.Normal(loc=self.weight_mu, scale=weight_sigma)
        bias_q = D.Normal(loc=self.bias_mu, scale=bias_sigma)
        
        # Create prior distributions (broadcast to match shape)
        device = self.weight_mu.device
        weight_prior = D.Normal(
            loc=torch.zeros_like(self.weight_mu, device=device) + self.prior_mu,
            scale=torch.ones_like(self.weight_mu, device=device) * self.prior_sigma
        )
        bias_prior = D.Normal(
            loc=torch.zeros_like(self.bias_mu, device=device) + self.prior_mu,
            scale=torch.ones_like(self.bias_mu, device=device) * self.prior_sigma
        )
        
        # Compute KL divergence using torch.distributions (like BayesianDeepWine notebook)
        weight_kl = D.kl.kl_divergence(weight_q, weight_prior).sum()
        bias_kl = D.kl.kl_divergence(bias_q, bias_prior).sum()
        
        return weight_kl + bias_kl


class BayesianMLP(nn.Module):
    """
    Bayesian Multi-Layer Perceptron using BayesByBackprop.
    
    This is a fully Bayesian neural network where all weights and biases
    are treated as probability distributions.
    
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
    activation : str
        Activation function ('relu', 'tanh', 'sigmoid', default: 'relu')
    prior_mu : float
        Prior mean (default: 0.0)
    prior_sigma : float
        Prior standard deviation (default: 1.0)
    posterior_mu_init : float
        Initial posterior mean (default: 0.0)
    posterior_rho_init : float
        Initial posterior rho (default: -3.0)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_units: int = 10,
        n_layers: int = 2,
        activation: str = 'relu',
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0
    ):
        super(BayesianMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(
            BayesianLinear(
                input_dim,
                hidden_units,
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init
            )
        )
        layers.append(self.activation)
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(
                BayesianLinear(
                    hidden_units,
                    hidden_units,
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    posterior_mu_init=posterior_mu_init,
                    posterior_rho_init=posterior_rho_init
                )
            )
            layers.append(self.activation)
        
        # Output layer
        layers.append(
            BayesianLinear(
                hidden_units,
                output_dim,
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                posterior_mu_init=posterior_mu_init,
                posterior_rho_init=posterior_rho_init
            )
        )
        
        self.layers = nn.ModuleList(layers)
        
        # Store Bayesian layers separately for KL divergence calculation
        self.bayesian_layers = [layer for layer in self.layers if isinstance(layer, BayesianLinear)]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Bayesian network.
        
        Each forward pass samples new weights from the posterior distribution,
        so the output is stochastic.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute total KL divergence across all Bayesian layers.
        
        Returns:
        --------
        torch.Tensor
            Total KL divergence (scalar)
        """
        total_kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.bayesian_layers:
            total_kl = total_kl + layer.kl_divergence()
        return total_kl


def compute_elbo_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    n_samples: int = 1,
    complexity_cost_weight: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ELBO (Evidence Lower BOund) loss for Bayesian neural network.
    
    ELBO = -log p(y|x,w) + KL(q(w|θ) || p(w))
    
    Where:
    - -log p(y|x,w) is the negative log-likelihood (data fit term)
    - KL(q(w|θ) || p(w)) is the KL divergence (complexity/regularization term)
    
    Parameters:
    -----------
    model : nn.Module
        Bayesian neural network model
    inputs : torch.Tensor
        Input features
    labels : torch.Tensor
        Target labels
    criterion : nn.Module
        Loss function (e.g., nn.MSELoss())
    n_samples : int
        Number of Monte Carlo samples for expectation (default: 1)
        More samples = better approximation but slower
    complexity_cost_weight : float
        Weight for KL divergence term (default: 1.0)
        Often set to 1/N where N is number of training samples
    
    Returns:
    --------
    elbo : torch.Tensor
        Total ELBO loss
    likelihood : torch.Tensor
        Negative log-likelihood (data fit term)
    kl_div : torch.Tensor
        KL divergence (complexity term)
    """
    # Monte Carlo sampling for expectation
    likelihood_samples = []
    
    for _ in range(n_samples):
        # Forward pass (samples weights from posterior)
        outputs = model(inputs)
        
        # Compute negative log-likelihood
        if isinstance(criterion, nn.MSELoss):
            # For regression: -log p(y|x,w) = MSE
            nll = criterion(outputs, labels)
        else:
            # For classification: -log p(y|x,w) = CrossEntropy
            nll = criterion(outputs, labels)
        
        likelihood_samples.append(nll)
    
    # Average over samples (Monte Carlo estimate)
    likelihood = torch.stack(likelihood_samples).mean()
    
    # KL divergence (complexity term)
    if hasattr(model, 'kl_divergence'):
        kl_div = model.kl_divergence()
    else:
        # If model doesn't have kl_divergence method, try to compute from layers
        kl_div = torch.tensor(0.0, device=inputs.device)
        for module in model.modules():
            if isinstance(module, BayesianLinear):
                kl_div = kl_div + module.kl_divergence()
    
    # ELBO = likelihood + complexity_weight * KL
    elbo = likelihood + complexity_cost_weight * kl_div
    
    return elbo, likelihood, kl_div


class ModelBuilder:
    """Builder class for creating Bayesian neural network architectures."""
    
    @staticmethod
    def build_bayesian_mlp(
        input_dim: int,
        output_dim: int = 1,
        hidden_units: int = 10,
        n_layers: int = 2,
        activation: str = 'relu',
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0
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
        activation : str
            Activation function
        prior_mu : float
            Prior mean
        prior_sigma : float
            Prior standard deviation
        posterior_mu_init : float
            Initial posterior mean
        posterior_rho_init : float
            Initial posterior rho
        
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
            activation=activation,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init
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


# Default model configurations using standard parameters from literature
# Based on Blundell et al. (2015) and common practices
DEFAULT_CONFIGS = {
    'bayesian_mlp': {
        'output_dim': 1,
        'hidden_units': 10,
        'n_layers': 2,
        'activation': 'relu',
        'prior_mu': 0.0,           # Standard: zero-mean prior
        'prior_sigma': 1.0,         # Standard: unit variance prior
        'posterior_mu_init': 0.0,   # Standard: initialize mean near zero
        'posterior_rho_init': -3.0  # Standard: gives sigma ≈ 0.05 initially
    },
    'bmlp': {  # Alias
        'output_dim': 1,
        'hidden_units': 10,
        'n_layers': 2,
        'activation': 'relu',
        'prior_mu': 0.0,
        'prior_sigma': 1.0,
        'posterior_mu_init': 0.0,
        'posterior_rho_init': -3.0
    }
}


# Example usage and testing
if __name__ == "__main__":
    # Create a simple Bayesian MLP
    model = BayesianMLP(
        input_dim=5,
        output_dim=1,
        hidden_units=10,
        n_layers=2
    )
    
    # Test forward pass
    x = torch.randn(32, 5)  # batch_size=32, features=5
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test KL divergence
    kl = model.kl_divergence()
    print(f"KL Divergence: {kl.item():.4f}")
    
    # Test ELBO loss
    labels = torch.randn(32, 1)
    criterion = nn.MSELoss()
    elbo, likelihood, kl_div = compute_elbo_loss(
        model, x, labels, criterion,
        n_samples=5,
        complexity_cost_weight=1.0 / 32
    )
    print(f"ELBO: {elbo.item():.4f}")
    print(f"Likelihood (MSE): {likelihood.item():.4f}")
    print(f"KL Divergence: {kl_div.item():.4f}")

