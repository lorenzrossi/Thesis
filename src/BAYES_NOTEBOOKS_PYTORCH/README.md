# BAYES_NOTEBOOKS_PYTORCH - Bayesian Neural Networks for Energy Forecasting

This folder contains modular PyTorch implementations of Bayesian neural networks for time series forecasting of Italian energy generation data with uncertainty quantification.

## Overview

Bayesian neural networks provide uncertainty estimates in addition to point predictions, making them valuable for risk assessment in energy forecasting. This implementation uses variational inference via the `blitz-bayesian-pytorch` library.

## Key Features

- **Uncertainty Quantification**: Provides confidence intervals for predictions
- **Variational Inference**: Efficient approximation of posterior weight distributions
- **ELBO Optimization**: Balances data fit and model complexity
- **Monte Carlo Sampling**: Multiple forward passes for robust predictions
- **Rolling Window Training**: Online learning approach for time series

## File Structure

### Core Modules (Blitz Implementation)
- **`models.py`**: Bayesian MLP model architecture definitions using blitz library
- **`trainer.py`**: Training and testing logic with uncertainty quantification (blitz)
- **`example_usage.py`**: Example scripts showing different usage patterns (blitz)

### Manual BayesByBackprop Implementation
Located in **`manual_bayesbybackprop/`** subfolder:
- **`models_bayesbybackprop.py`**: Manual implementation of BayesByBackprop from scratch
  - Uses `torch.distributions` (like BayesianDeepWine notebook)
  - `BayesianLinear`: Manual Bayesian linear layer with reparameterization trick
  - `BayesianMLP`: Bayesian MLP using manual layers
  - `compute_elbo_loss`: Manual ELBO calculation
  - Standard parameters from literature (Blundell et al. 2015)
- **`trainer_bayesbybackprop.py`**: Trainer using manual implementation
- **`example_usage_manual.py`**: Examples for manual implementation
- **`README.md`**: Detailed documentation for manual implementation

### Notebooks (Original)
- **`BMLP.ipynb`**: Bayesian MLP with time series only (lag_1)
- **`BMLP_WE.ipynb`**: Bayesian MLP with weekend dummy variables
- **`BMLP_BH.ipynb`**: Bayesian MLP with business hour dummy
- **`LSTM_pytorch_struttura.ipynb`**: Standard LSTM template (not Bayesian)

### Documentation
- **`ANALYSIS.md`**: Detailed analysis of the notebooks
- **`README.md`**: This file

## Installation

### Required Dependencies

```bash
pip install torch blitz-bayesian-pytorch numpy pandas scikit-learn matplotlib
```

Or install from requirements.txt:
```bash
pip install -r ../../requirements.txt
```

## Quick Start

### Option 1: Train Single Model

```python
from trainer import BayesianModelTrainer
from data_preprocessing import preprocess_pipeline

# Load data
data = preprocess_pipeline(
    data_path="data",
    download_from_drive=False,
    years=[2019, 2020, 2021]
)

# Create trainer
trainer = BayesianModelTrainer(
    feature_type='ts_only',  # or 'weekend', 'business_hour'
    window_size=1,  # Use lag_1
    n_forecast_steps=24,
    epochs=5,
    batch_size=168,
    learning_rate=0.01,
    device='mps'  # or 'cpu', 'cuda', None for auto-detect
)

# Train model
results = trainer.train(data, n_train=35064)

# Access results
print(f"RMSE: {results['overall_rmse']:.2f}")
print(f"Coverage: {results['coverage']:.2%}")  # % of true values within CI
print(f"Predictions: {results['predictions']}")
print(f"Confidence Intervals: {results['ci_upper']}, {results['ci_lower']}")
```

### Option 2: Run All Configurations

```python
from trainer import run_all_models

results_df = run_all_models(
    data_path="data",
    n_train=35064,
    save_results=True,
    results_path="results"
)

print(results_df)
```

## Architecture

### Bayesian MLP

The Bayesian MLP uses variational layers from the `blitz` library:

- **Input Layer**: BayesianLinear(input_dim → 10)
- **Hidden Layer**: BayesianLinear(10 → 10)
- **Output Layer**: BayesianLinear(10 → 1)

**Prior Settings:**
- `prior_sigma_1=0.01`
- `prior_sigma_2=0.01`
- `prior_pi=1.0`

### ELBO Loss

The training uses Evidence Lower BOund (ELBO) which consists of:
1. **ELBO**: Overall variational objective
2. **MSE**: Data fit term
3. **KL Divergence**: Regularization term (complexity cost)

## Feature Types

### 1. Time Series Only (`ts_only`)
- Uses `lag_1` only
- Input dimension: 1
- Simplest configuration

### 2. Weekend Dummies (`weekend`)
- Uses `lag_1` + `saturday` + `sunday`
- Input dimension: 3
- Captures weekly patterns

### 3. Business Hour (`business_hour`)
- Uses `lag_1` + `business hour`
- Input dimension: 2
- Captures daily business patterns

## Uncertainty Quantification

### Monte Carlo Sampling

During inference, the model samples multiple times from the weight distribution:

```python
# Default: 100 samples
preds = [model(X) for _ in range(100)]
preds = torch.stack(preds)

# Calculate statistics
means = preds.mean(axis=0)
stds = preds.std(axis=0)
```

### Confidence Intervals

Confidence intervals are calculated using:
```python
ci_upper = means + (std_multiplier * stds)  # Default: 5σ
ci_lower = means - (std_multiplier * stds)
```

### Coverage

The `coverage` metric indicates what percentage of true values fall within the confidence intervals.

## Training Strategy

### Rolling Window Approach

1. **Initial Training**: Train on first `n_train` samples
2. **Forecast**: Predict next `n_forecast_steps` (default: 24 hours)
3. **Update**: Remove oldest data, add observed values
4. **Retrain**: Train on updated window
5. **Repeat**: Continue for all test periods

### Key Parameters

- `n_train`: Training window size (default: 35064 = 3 years)
- `n_forecast_steps`: Forecast horizon (default: 24 hours)
- `epochs`: Training epochs per window (default: 5)
- `sample_nbr`: ELBO samples during training (default: 5)
- `inference_samples`: Monte Carlo samples for prediction (default: 100)

## Results Dictionary

The `train()` method returns a dictionary with:

```python
{
    'model_type': 'bayesian_mlp',
    'feature_type': 'ts_only',
    'predictions': np.ndarray,      # Point predictions
    'truth': np.ndarray,             # True values
    'ci_upper': np.ndarray,          # Upper confidence bounds
    'ci_lower': np.ndarray,          # Lower confidence bounds
    'stds': np.ndarray,              # Prediction standard deviations
    'overall_rmse': float,          # Overall RMSE
    'overall_mae': float,           # Overall MAE
    'overall_r2': float,            # Overall R²
    'coverage': float,               # Coverage percentage
    'period_errors': List[float],   # RMSE per period
    'n_periods': int                # Number of forecast periods
}
```

## Visualization

### Plotting Predictions with Uncertainty

```python
import matplotlib.pyplot as plt

results = trainer.train(data, n_train=35064)

plt.figure(figsize=(15, 6))
plt.plot(results['truth'], label='Actual', color='blue')
plt.plot(results['predictions'], label='Predicted', color='red')
plt.fill_between(
    range(len(results['predictions'])),
    results['ci_lower'],
    results['ci_upper'],
    alpha=0.3,
    color='orange',
    label='Confidence Interval'
)
plt.legend()
plt.title('Bayesian MLP Predictions with Uncertainty')
plt.xlabel('Time')
plt.ylabel('Total Aggregated Energy (MW)')
plt.show()
```

## Comparison with Frequentist Models

### Advantages
- ✅ **Uncertainty Quantification**: Provides confidence intervals
- ✅ **Regularization**: Built-in through KL divergence
- ✅ **Robustness**: Less prone to overfitting
- ✅ **Risk Assessment**: Can quantify prediction uncertainty

### Disadvantages
- ❌ **Computational Cost**: More expensive (multiple forward passes)
- ❌ **Complexity**: Harder to implement and understand
- ❌ **Hyperparameter Sensitivity**: Prior settings matter
- ❌ **Library Dependency**: Requires `blitz` library

## Device Support

The trainer automatically detects and uses:
1. **CUDA** (NVIDIA GPU) - if available
2. **MPS** (Apple Silicon GPU) - if available
3. **CPU** - fallback

```python
trainer = BayesianModelTrainer(
    feature_type='ts_only',
    device='mps'  # Explicitly set, or None for auto-detect
)
```

## Example Usage

See `example_usage.py` for complete examples of:
- Training single models
- Using different feature types
- Running all configurations
- Accessing uncertainty information

## Command Line Usage

```bash
# Train all configurations
python trainer.py --features all --data_path data

# Train specific configuration
python trainer.py --features weekend --n_train 35064 --device mps

# Download from Google Drive
python trainer.py --download --folder_id "1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H"
```

## Troubleshooting

### Import Error: blitz not found
```bash
pip install blitz-bayesian-pytorch
```

### MPS not available
- Requires macOS 12.3+ (Monterey or later)
- Check: `torch.backends.mps.is_available()`
- Falls back to CPU if not available

### Low Coverage
- Adjust `std_multiplier` (default: 5.0)
- Increase `inference_samples` for more stable estimates
- Check if model is properly trained

## Manual BayesByBackprop Implementation

For educational purposes and full control, we also provide a **manual implementation** of BayesByBackprop:

### Using Manual Implementation

```python
from trainer_bayesbybackprop import BayesianModelTrainer
from data_preprocessing import preprocess_pipeline

# Load data
data = preprocess_pipeline(data_path="data", download_from_drive=False)

# Create trainer (uses manual implementation)
trainer = BayesianModelTrainer(
    feature_type='ts_only',
    window_size=1,
    epochs=5,
    device='mps'
)

# Train (same interface as blitz version)
results = trainer.train(data, n_train=35064)
```

### Understanding the Implementation

The manual implementation (`models_bayesbybackprop.py`) shows:
- **Reparameterization Trick**: How weights are sampled
- **KL Divergence**: Manual calculation
- **ELBO Loss**: Complete implementation
- **Standard Parameters**: From Blundell et al. (2015)

See `example_usage_manual.py` for detailed examples.

### When to Use Manual vs Blitz

- **Use Blitz** (`models.py`, `trainer.py`): Production, quick prototyping
- **Use Manual** (`models_bayesbybackprop.py`, `trainer_bayesbybackprop.py`): Learning, research, customization

Both implement the same algorithm - choose based on your needs!

## References

- **Blitz Library**: [blitz-bayesian-pytorch](https://github.com/piEsposito/blitz-bayesian-pytorch)
- **Variational Inference**: [Blundell et al., 2015](https://arxiv.org/abs/1505.05424) - Original BayesByBackprop paper
- **Comparison Document**: See `BLITZ_VS_BAYESBYBACKPROP.md` for detailed comparison

## Next Steps

1. **Create Unified Notebook**: Similar to FREQ_NETS_TORCH
2. **Add Bayesian LSTM**: Extend to recurrent architectures
3. **Hyperparameter Tuning**: Optimize prior settings
4. **Comparison Analysis**: Compare with frequentist models
5. **Study Manual Implementation**: Understand BayesByBackprop internals

