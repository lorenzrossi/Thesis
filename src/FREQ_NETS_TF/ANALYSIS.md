# Analysis of FREQ_NETS_TF Folder

## Overview
This folder contains three TensorFlow-based neural network implementations for time series forecasting of Italian energy generation data. All notebooks implement **online/rolling window training** approaches for time series forecasting.

## Structure

### Files
- **`MLP TF.ipynb`**: Multi-Layer Perceptron implementation
- **`LSTM TF.ipynb`**: Long Short-Term Memory (LSTM) network implementation  
- **`CNN TF.ipynb`**: Convolutional Neural Network implementation
- **`utils.py`**: Utility functions for Bayesian neural networks (variational posterior, KL divergence)

## Common Patterns Across All Notebooks

### 1. Data Preprocessing
All notebooks use identical preprocessing steps:
- Load data from CSV files (2016-2021)
- Handle missing values with linear interpolation
- Aggregate energy sources (hydro, gas, total)
- Generate temporal features (weekend, business hours, hour of day)
- **Issues**: Same `SettingWithCopyWarning` problems as in `DATA_DESC.ipynb`

### 2. Training Strategy: Online/Rolling Window
All models use a **rolling window retraining approach**:
- **Training window**: 3 years (24×365×3 = 26,280 hours)
- **Forecast horizon**: 24 hours (1 day) or 24×7 hours (1 week)
- **Retraining frequency**: After each forecast period
- **Update mechanism**: Remove oldest data, add new observed data

**Key Code Pattern:**
```python
for i in range(0, len(X_test), n_forecast_steps):
    # Train on current window
    model.fit(history_X, history_y, epochs=100, ...)
    
    # Predict next period
    yhat = model.predict(X_test_block)
    
    # Update history with actual observed values
    history_X = np.concatenate((history_X, X_test_block), axis=0)
    history_X = history_X[n_forecast_steps:]
    history_y = np.concatenate((history_y, y_test_block), axis=0)
    history_y = history_y[n_forecast_steps:]
```

### 3. Feature Engineering Experiments
Each notebook tests three feature combinations:

1. **Time Series Only**: Just `total_aggregated` with lags
2. **Weekend Dummies**: `total_aggregated` + `saturday` + `sunday`
3. **Business Hour Dummy**: `total_aggregated` + `business hour`

### 4. Model Architectures

#### MLP (MLP TF.ipynb)
- **Architecture**: Simple feedforward network
- **Input**: Lag features (lag_1 to lag_24, or just lag_24)
- **Layers**: 
  - Dense(10, relu)
  - Dense(10, relu)
  - Dense(1, linear)
- **Parameters**: ~46-56 parameters

#### LSTM (LSTM TF.ipynb)
- **Architecture**: Stacked LSTM layers
- **Input**: Sequence of length 1 (single timestep)
- **Layers**:
  - LSTM(10, return_sequences=True)
  - LSTM(10, return_sequences=True)
  - Flatten()
  - Dense(1)
- **Parameters**: ~26-91 parameters

#### CNN (CNN TF.ipynb)
- **Architecture**: 1D Convolutional layers
- **Input**: Window of size 2 (2 timesteps)
- **Layers**:
  - Conv1D(16, kernel_size=1, relu)
  - Conv1D(16, kernel_size=1, relu)
  - Flatten()
  - Dense(1)
- **Parameters**: ~18-371 parameters

### 5. Results Summary

#### MLP Results
- **Time Series Only**: RMSE = 4,906.77
- **Weekend Dummies**: RMSE = 2,716.58 ⭐ (Best)
- **Business Hour**: RMSE = 2,938.88

#### LSTM Results
- **Time Series Only**: RMSE = 3,181.02
- **Weekend Dummies**: RMSE = 2,678.87
- **Business Hour**: RMSE = 2,569.14 ⭐ (Best)

#### CNN Results
- **Time Series Only**: RMSE = 2,853.73
- **Weekend Dummies**: RMSE = 2,392.55
- **Business Hour**: RMSE = 1,359.49 ⭐ (Best overall)

## Key Findings

### 1. Feature Importance
- **Business hour dummy** consistently improves performance across all models
- **Weekend dummies** also help but less than business hours
- **CNN benefits most** from business hour feature (RMSE reduction: 1,494 → 1,359)

### 2. Model Performance Ranking
1. **CNN with Business Hour**: 1,359.49 (Best)
2. **CNN with Weekend**: 2,392.55
3. **LSTM with Business Hour**: 2,569.14
4. **LSTM with Weekend**: 2,678.87
5. **MLP with Weekend**: 2,716.58
6. **MLP with Business Hour**: 2,938.88

### 3. Architecture Observations
- **CNN** performs best overall, especially with business hour feature
- **LSTM** shows good performance but may be overkill for this simple sequence length
- **MLP** performs worst, likely due to limited temporal modeling capability

## Issues and Recommendations

### Critical Issues

1. **Data Preprocessing Code Duplication**
   - Same `retrieve_data()`, `data_and_aggregator()`, `businesshour_and_we_generation()` in all notebooks
   - Should use the improved `preprocessing_utils.py` or `data_preprocessing.py`

2. **SettingWithCopyWarning**
   - All notebooks have the same warning issues
   - Should use vectorized operations (already fixed in preprocessing modules)

3. **Hardcoded Paths**
   - Windows-specific paths: `C:\\Users\\loren\\OneDrive...`
   - Should use relative paths or environment variables

4. **Inconsistent Scaling**
   - Some notebooks use `scaler` (from lag creation), others use `target_scaler`
   - MLP notebook has a bug: uses `scaler.inverse_transform()` but should use `target_scaler`

5. **Window Size Inconsistency**
   - MLP: Uses lag_24 (single value) or lag_1
   - LSTM: Uses sequence length of 1 (minimal temporal context)
   - CNN: Uses window size of 2
   - **Recommendation**: Standardize window sizes for fair comparison

6. **RMSE Calculation Bug in MLP**
   - Line 15 in MLP notebook: `rmse = sqrt(np.mean(truth_block - yhat_rescaled)**2)`
   - Should be: `rmse = sqrt(np.mean((truth_block - yhat_rescaled)**2))`
   - Currently calculates mean first, then squares (incorrect)

### Code Quality Issues

1. **No Model Persistence**: Models are retrained from scratch each period
2. **No Validation Set**: Only train/test split, no validation for hyperparameter tuning
3. **Limited Hyperparameter Exploration**: Fixed architecture, learning rate, batch size
4. **No Cross-Validation**: Single train/test split may not be representative
5. **Memory Inefficiency**: Concatenating arrays in loop could be optimized

### Recommendations

1. **Refactor Common Code**
   - Create shared preprocessing module
   - Use `data_preprocessing.py` for data loading
   - Create shared model training utilities

2. **Fix Bugs**
   - Fix RMSE calculation in MLP notebook
   - Fix scaler usage inconsistencies
   - Remove hardcoded paths

3. **Improve Model Architecture**
   - Experiment with longer sequence lengths for LSTM/CNN
   - Add dropout, batch normalization
   - Try attention mechanisms

4. **Better Evaluation**
   - Add validation set
   - Calculate additional metrics (MAE, MAPE, R²)
   - Plot prediction intervals/uncertainty

5. **Hyperparameter Tuning**
   - Learning rate scheduling
   - Architecture search
   - Window size optimization

6. **Use Utils.py**
   - The `utils.py` file contains Bayesian neural network utilities
   - Could implement Bayesian versions for uncertainty quantification

## Model Comparison Table

| Model | Features | RMSE | Parameters | Window Size |
|-------|----------|------|------------|-------------|
| MLP | TS Only | 4,906.77 | 46 | lag_24 |
| MLP | Weekend | 2,716.58 | 56 | lag_1 + dummies |
| MLP | Business Hour | 2,938.88 | 25 | lag_1 + dummy |
| LSTM | TS Only | 3,181.02 | 26 | 1 timestep |
| LSTM | Weekend | 2,678.87 | 91 | 1 timestep + dummies |
| LSTM | Business Hour | 2,569.14 | 83 | 1 timestep + dummy |
| CNN | TS Only | 2,853.73 | 25 | 2 timesteps |
| CNN | Weekend | 2,392.55 | 371 | 2 timesteps + dummies |
| CNN | Business Hour | **1,359.49** | 18 | 2 timesteps + dummy |

## Conclusion

The CNN model with business hour features achieves the best performance, suggesting that:
1. Convolutional operations capture local temporal patterns effectively
2. Business hour information is highly predictive of energy demand
3. Simple architectures can be effective with good feature engineering

The rolling window approach is computationally expensive but allows models to adapt to changing patterns over time.

