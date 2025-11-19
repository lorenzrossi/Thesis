# Analysis of BAYES_NOTEBOOKS_PYTORCH Folder

## Overview
This folder contains Bayesian neural network implementations for time series forecasting of Italian energy generation data using PyTorch. The notebooks implement Bayesian Multi-Layer Perceptrons (BMLP) with different feature combinations and a standard LSTM structure.

## Structure

### Files
- **`BMLP.ipynb`**: Bayesian MLP with time series only features (lag_1)
- **`BMLP_WE.ipynb`**: Bayesian MLP with weekend dummy variables (saturday, sunday)
- **`BMLP_BH.ipynb`**: Bayesian MLP with business hour dummy variable
- **`LSTM_pytorch_struttura.ipynb`**: Standard (non-Bayesian) LSTM structure/template

## Common Patterns Across Notebooks

### 1. Data Preprocessing
All BMLP notebooks use identical preprocessing steps:
- Load data from CSV files (2016-2021)
- Handle missing values (implicitly through bfill)
- Aggregate energy sources (hydro, gas, total)
- Generate temporal features (weekend, business hours, hour of day)

**Issues:**
- Same `SettingWithCopyWarning` problems as in other notebooks
- Hardcoded Windows paths: `os.chdir('C:\\Users\\loren\\...')`
- Inefficient iteration in `retrieve_data()` and `businesshour_and_we_generation()`
- Uses `bfill()` for missing values instead of proper interpolation

### 2. Bayesian Neural Network Architecture
All BMLP notebooks use the same architecture:
- **Library**: `blitz` (Bayesian Layers in Torch)
- **Architecture**: 2-layer Bayesian MLP
  - Input → 10 units (BayesianLinear)
  - 10 units → 10 units (BayesianLinear)
  - 10 units → 1 output (BayesianLinear)
- **Prior settings**: `prior_sigma_1=0.01, prior_sigma_2=0.01, prior_pi=1`
- **Decorator**: `@variational_estimator` for automatic ELBO calculation

### 3. Training Strategy: Rolling Window
All notebooks use a **rolling window retraining approach**:
- **Training window**: Varies (e.g., 24*365 = 8,760 hours for 1 year)
- **Forecast horizon**: 24 hours
- **Retraining frequency**: After each forecast period
- **Update mechanism**: Remove oldest data, add new observed data

**Key Code Pattern:**
```python
for i in range(0, len(X_test), n_forecast_steps):
    # Train on current window
    train_model(regressor, dataloader_train, optimizer, criterion)
    
    # Predict next period with uncertainty
    y_pred, ci_upper, ci_lower, stds = evaluate_regression(regressor, X_test_block)
    
    # Update history with actual observed values
    # (implementation varies)
```

### 4. Feature Engineering Experiments
Each notebook tests different feature combinations:

1. **BMLP.ipynb - Time Series Only**: 
   - Uses `lag_1` only (single lag feature)
   - Input dimension: 1

2. **BMLP_WE.ipynb - Weekend Dummies**:
   - Uses `lag_1` + `saturday` + `sunday`
   - Input dimension: 3

3. **BMLP_BH.ipynb - Business Hour Dummy**:
   - Uses `lag_1` + `business hour`
   - Input dimension: 2

### 5. Bayesian-Specific Features

#### Uncertainty Quantification
- **Monte Carlo Sampling**: Uses 100 samples for prediction
- **Confidence Intervals**: Calculates upper/lower bounds using `std_multiplier=5`
- **Prediction Distribution**: Mean and standard deviation from multiple forward passes

#### ELBO Loss
- **Components**: 
  1. ELBO (Evidence Lower BOund)
  2. MSE (Mean Squared Error)
  3. KL Divergence (regularization term)
- **Complexity Cost**: Weighted by `1./X_train.shape[0]`
- **Sample Number**: 5 samples per forward pass during training

#### Evaluation Function
```python
def evaluate_regression(regressor, X, y, samples=100, std_multiplier=5):
    # Sample predictions multiple times
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    
    # Calculate statistics
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    
    # Confidence intervals
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    
    return y_pred, y_true, upper_inv, lower_inv, stds
```

## Issues and Bugs

### 1. **Hardcoded Windows Paths**
**Location**: All notebooks, Cell 1
```python
os.chdir('C:\\Users\\loren\\OneDrive - Università degli Studi di Milano\\Lezioni uni\\Tesi\\Dataset Energia\\')
```
**Issue**: Not portable, breaks on other systems
**Fix**: Use relative paths or environment variables

### 2. **SettingWithCopyWarning**
**Location**: `retrieve_data()` and `businesshour_and_we_generation()` functions
**Issue**: Iterating over DataFrame and modifying in place
**Fix**: Use vectorized operations (already fixed in `data_preprocessing.py`)

### 3. **Inefficient Data Loading**
**Location**: `retrieve_data()` function
```python
for i, row in df.iterrows():
    df["MTU"][i] = df['MTU'][i][:16]
```
**Issue**: Row-by-row iteration is very slow
**Fix**: Vectorized string operations: `df['MTU'] = df['MTU'].astype(str).str[:16]`

### 4. **Inconsistent Scaling**
**Location**: Different notebooks use different scalers
- Some use `StandardScaler` for both X and y
- Some use separate scalers
- Some reuse the same scaler (incorrect)

### 5. **Missing Value Handling**
**Location**: After creating lags
```python
df = df.bfill()  # Only backward fill
```
**Issue**: Should use proper interpolation (linear, forward, backward)
**Fix**: Use `df.interpolate(method='linear')` or the `handle_missing_values()` function

### 6. **Device Management**
**Location**: LSTM notebook
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
**Issue**: Doesn't check for MPS (Apple Silicon GPU)
**Fix**: Add MPS detection: `'mps' if torch.backends.mps.is_available() else 'cpu'`

### 7. **Inconsistent Training Window Size**
**Location**: Different notebooks
- BMLP.ipynb: `n_train = 24*365` (1 year)
- Some use `n_train = 35064` (3 years)
- Should be consistent across experiments

### 8. **Commented Code**
**Location**: Throughout notebooks
**Issue**: Lots of commented-out code makes notebooks harder to read
**Fix**: Remove or document why code is commented

## Differences Between Notebooks

### BMLP.ipynb (Time Series Only)
- **Features**: `lag_1` only
- **Input size**: 1
- **Simplest configuration**

### BMLP_WE.ipynb (Weekend Dummies)
- **Features**: `lag_1`, `saturday`, `sunday`
- **Input size**: 3
- **Captures weekly patterns**

### BMLP_BH.ipynb (Business Hour Dummy)
- **Features**: `lag_1`, `business hour`
- **Input size**: 2
- **Captures daily business patterns**

### LSTM_pytorch_struttura.ipynb
- **Not Bayesian**: Standard LSTM implementation
- **Structure**: 2-layer LSTM with dropout
- **Purpose**: Appears to be a template/structure reference
- **Incomplete**: Missing data loading and full training loop

## Key Features of Bayesian Implementation

### 1. Uncertainty Quantification
- Provides prediction intervals, not just point estimates
- Quantifies epistemic uncertainty (model uncertainty)
- Useful for risk assessment in energy forecasting

### 2. Variational Inference
- Uses Variational Bayes (VB) approach
- Approximates posterior distribution of weights
- More efficient than MCMC for large models

### 3. ELBO Optimization
- Balances data fit (MSE) and model complexity (KL divergence)
- Prevents overfitting through regularization
- Complexity cost adapts to training set size

### 4. Monte Carlo Sampling
- Multiple forward passes during inference
- Each pass samples from weight distribution
- Aggregates samples to get prediction distribution

## Recommendations

### 1. **Modularization** (High Priority)
- Extract common functions to `models.py` and `trainer.py`
- Create `BayesianMLP` class similar to frequentist models
- Reuse `data_preprocessing.py` instead of duplicating code

### 2. **Fix Data Loading** (High Priority)
- Replace hardcoded paths with flexible paths
- Use `data_preprocessing.py` functions
- Fix `SettingWithCopyWarning` issues

### 3. **Standardize Training** (Medium Priority)
- Consistent training window size
- Standardize scaler usage
- Unified rolling window implementation

### 4. **Improve Documentation** (Medium Priority)
- Add markdown cells explaining Bayesian concepts
- Document ELBO components
- Explain uncertainty quantification approach

### 5. **Create Unified Notebook** (Low Priority)
- Similar to `Unified_Training_Notebook.ipynb` in FREQ_NETS_TORCH
- Combine all three BMLP variants
- Add comparison visualizations

### 6. **Add MPS Support** (Low Priority)
- Update device detection for Apple Silicon
- Test on MPS if available

### 7. **Code Cleanup** (Low Priority)
- Remove commented code
- Consistent naming conventions
- Add type hints

## Comparison with Frequentist Models

### Advantages of Bayesian Approach
1. **Uncertainty Quantification**: Provides confidence intervals
2. **Regularization**: Built-in through KL divergence
3. **Robustness**: Less prone to overfitting
4. **Interpretability**: Can analyze weight distributions

### Disadvantages
1. **Computational Cost**: More expensive (multiple forward passes)
2. **Complexity**: Harder to implement and understand
3. **Hyperparameter Sensitivity**: Prior settings matter
4. **Library Dependency**: Requires `blitz` library

## Dependencies

### Required Libraries
- `torch`: PyTorch
- `blitz`: Bayesian Layers in Torch (`pip install blitz-bayesian-pytorch`)
- Standard ML libraries: `numpy`, `pandas`, `sklearn`, `matplotlib`

### Missing from requirements.txt
- `blitz-bayesian-pytorch` should be added
- `statsmodels` (used in some notebooks)
- `seaborn` (used for visualization)

## Next Steps

1. **Create modular structure** similar to FREQ_NETS_TORCH
2. **Fix data loading** to use `data_preprocessing.py`
3. **Standardize training** procedures
4. **Add comprehensive README** explaining Bayesian approach
5. **Create unified training notebook** for all variants
6. **Update requirements.txt** with missing dependencies

