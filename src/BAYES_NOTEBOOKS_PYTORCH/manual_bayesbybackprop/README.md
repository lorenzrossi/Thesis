# Manual BayesByBackprop Implementation

This folder contains a **manual implementation** of BayesByBackprop from scratch, following the style of the [BayesianDeepWine notebook](https://colab.research.google.com/github/charlesollion/dlexperiments/blob/master/6-Bayesian-DL/BayesByBackprop_pytorch.ipynb).

## Overview

This implementation provides:
- **Full transparency**: See exactly how BayesByBackprop works
- **Educational value**: Learn the algorithm step-by-step
- **PyTorch-native**: Uses `torch.distributions` for clean code
- **No external dependencies**: Pure PyTorch (no blitz library needed)
- **Standard parameters**: From Blundell et al. (2015) literature

## Key Differences from Blitz Implementation

| Feature | Manual (This Folder) | Blitz (Parent Folder) |
|---------|---------------------|----------------------|
| **Implementation** | From scratch | Library-based |
| **Code Style** | Uses `torch.distributions` | Uses blitz decorators |
| **Reparameterization** | `q.rsample()` | Automatic via decorator |
| **KL Divergence** | `torch.distributions.kl.kl_divergence` | Automatic via decorator |
| **Dependencies** | Only PyTorch | Requires blitz library |
| **Customization** | Full control | Limited to library options |
| **Best For** | Learning, research | Production, prototyping |

## Files

- **`models_bayesbybackprop.py`**: Manual implementation of Bayesian layers
  - `BayesianLinear`: Bayesian linear layer using `torch.distributions`
  - `BayesianMLP`: Multi-layer Bayesian network
  - `compute_elbo_loss`: ELBO calculation function
  
- **`trainer_bayesbybackprop.py`**: Training and evaluation logic
  - Same interface as blitz trainer
  - Uses manual ELBO calculation
  - Full control over training process

- **`example_usage_manual.py`**: Usage examples and tutorials

## Implementation Style

This implementation follows the **BayesianDeepWine notebook** style:

### 1. Using `torch.distributions`

```python
import torch.distributions as D

# Create posterior distribution
q = D.Normal(loc=mu, scale=sigma)

# Sample using reparameterization trick
w = q.rsample()  # Differentiable!
```

### 2. KL Divergence Calculation

```python
# Create prior
prior = D.Normal(loc=0.0, scale=1.0)

# Compute KL divergence
kl = D.kl.kl_divergence(q, prior).sum()
```

### 3. ELBO Loss

```python
# ELBO = -log p(y|x,w) + KL(q(w|θ) || p(w))
elbo = mse_loss + (complexity_weight * kl)
```

## Standard Parameters

From Blundell et al. (2015):

```python
# Prior distribution
prior_mu = 0.0        # Zero-mean Gaussian
prior_sigma = 1.0     # Unit variance

# Posterior initialization
posterior_mu_init = 0.0    # Initialize mean near zero
posterior_rho_init = -3.0  # Gives initial sigma ≈ 0.05
```

## Quick Start

```python
from trainer_bayesbybackprop import BayesianModelTrainer
from data_preprocessing import preprocess_pipeline

# Load data
data = preprocess_pipeline(data_path="data", download_from_drive=False)

# Create trainer (manual implementation)
trainer = BayesianModelTrainer(
    feature_type='ts_only',
    window_size=1,
    epochs=5,
    device='mps'
)

# Train
results = trainer.train(data, n_train=35064)

# Results include uncertainty quantification
print(f"RMSE: {results['overall_rmse']:.2f}")
print(f"Coverage: {results['coverage']:.2%}")
```

## Understanding the Code

### BayesianLinear Layer

```python
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        # Learn mu (mean) and rho (log variance) for weights
        self.weight_mu = nn.Parameter(...)
        self.weight_rho = nn.Parameter(...)
        # Same for bias
    
    def forward(self, x):
        # Create posterior distribution
        q = D.Normal(loc=self.weight_mu, scale=sigma)
        
        # Sample weights (reparameterization trick)
        weight = q.rsample()
        
        # Forward pass
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        # Compute KL(q || prior)
        q = D.Normal(loc=mu, scale=sigma)
        prior = D.Normal(loc=0.0, scale=1.0)
        return D.kl.kl_divergence(q, prior).sum()
```

### ELBO Calculation

```python
def compute_elbo_loss(model, inputs, labels, criterion, n_samples=1, complexity_weight=1.0):
    # Monte Carlo sampling
    likelihood_samples = []
    for _ in range(n_samples):
        outputs = model(inputs)  # Samples weights
        nll = criterion(outputs, labels)
        likelihood_samples.append(nll)
    
    likelihood = torch.stack(likelihood_samples).mean()
    
    # KL divergence
    kl = model.kl_divergence()
    
    # ELBO
    elbo = likelihood + complexity_weight * kl
    return elbo, likelihood, kl
```

## Comparison with BayesianDeepWine Notebook

This implementation is **compatible** with the BayesianDeepWine notebook approach:

| Notebook Feature | Our Implementation |
|-----------------|-------------------|
| `torch.distributions.Normal` | ✅ Used |
| `q.rsample()` | ✅ Used |
| `torch.distributions.kl.kl_divergence` | ✅ Used |
| `torch.log(1. + torch.exp(rho))` | ✅ Used (softplus) |
| Unit Gaussian prior | ✅ Default |
| Reparameterization trick | ✅ Automatic via `rsample()` |

## Advantages

1. **Educational**: See exactly how BayesByBackprop works
2. **Transparent**: All implementation details visible
3. **Customizable**: Easy to modify priors, sampling, etc.
4. **No dependencies**: Pure PyTorch
5. **Research-ready**: Perfect for experiments

## When to Use

- ✅ **Learning**: Understanding how BayesByBackprop works
- ✅ **Research**: Experimenting with different priors/sampling
- ✅ **Customization**: Need to modify the algorithm
- ✅ **Education**: Teaching Bayesian neural networks

For production or quick prototyping, consider using the **blitz implementation** in the parent folder.

## References

- **Blundell et al. (2015)**: [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424)
- **BayesianDeepWine Notebook**: [Colab Tutorial](https://colab.research.google.com/github/charlesollion/dlexperiments/blob/master/6-Bayesian-DL/BayesByBackprop_pytorch.ipynb)

