# FREQ_NETS_TF - Frequency Domain Neural Networks (TensorFlow)

This folder contains implementations of MLP, LSTM, and CNN models for time series forecasting of Italian energy generation data.

## File Structure

### Core Modules
- **`models.py`**: Model architecture definitions (MLP, LSTM, CNN builders)
- **`trainer.py`**: Training and testing logic with rolling window approach
- **`utils.py`**: Bayesian neural network utilities (for future use)

### Notebooks (Original)
- **`MLP TF.ipynb`**: Original MLP notebook
- **`LSTM TF.ipynb`**: Original LSTM notebook
- **`CNN TF.ipynb`**: Original CNN notebook

### Documentation & Examples
- **`example_usage.py`**: Example scripts showing different usage patterns
- **`ANALYSIS.md`**: Detailed analysis of the notebooks
- **`README.md`**: This file

## Architecture

The code is split into two main modules:

### 1. `models.py` - Model Building
Contains model architecture definitions:
- `ModelBuilder`: Class with static methods to build models
- `create_model()`: Factory function to create models
- `DEFAULT_CONFIGS`: Default configurations matching original notebooks

**Supported Models:**
- **MLP**: Multi-Layer Perceptron (feedforward network)
- **LSTM**: Long Short-Term Memory (recurrent network)
- **CNN**: Convolutional Neural Network (1D convolutions)

### 2. `trainer.py` - Training & Testing
Contains training and evaluation logic:
- `DataPreparator`: Handles data preparation for different model types
- `ModelTrainer`: Main training class with rolling window approach
- `run_all_models()`: Function to run all model configurations

## Quick Start

### Option 1: Train Single Model

```python
from trainer import ModelTrainer
from data_preprocessing import preprocess_pipeline

# Load data
data = preprocess_pipeline(data_path="data", download_from_drive=False)

# Create trainer
trainer = ModelTrainer(
    model_type='cnn',
    feature_type='business_hour',
    window_size=2,
    verbose=True
)

# Train
results = trainer.train(data, n_train=35064)
print(f"RMSE: {results['overall_rmse']:.2f}")
```

### Option 2: Run All Models

```python
from trainer import run_all_models

# Run all 9 configurations (3 models × 3 feature types)
results = run_all_models(
    data_path="data",
    download_from_drive=False,
    n_train=35064
)
```

### Option 3: Build Custom Model

```python
from models import create_model, ModelBuilder

# Create a custom CNN model
model = create_model(
    'cnn',
    input_shape=(2, 2),  # (timesteps, features)
    filters=32,  # More filters
    n_layers=3,  # More layers
    dropout_rate=0.2  # Add dropout
)

# Or use ModelBuilder directly
builder = ModelBuilder()
model = builder.build_cnn(
    input_shape=(2, 2),
    filters=32,
    n_layers=3,
    dropout_rate=0.2
)
builder.compile_model(model, learning_rate=0.001)
```

### Option 4: Command Line

```bash
# Run all models
python trainer.py --model all --features all

# Run specific model
python trainer.py --model cnn --features business_hour

# Download from Google Drive
python trainer.py --download --folder_id "1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H"
```

## Model Configurations

### Model Types
- **MLP**: Multi-Layer Perceptron
  - Default: 2 hidden layers, 10 units each
  - Window size: 24 lags (or 1 for dummies)
  
- **LSTM**: Long Short-Term Memory
  - Default: 2 LSTM layers, 10 units each
  - Window size: 1 timestep
  
- **CNN**: Convolutional Neural Network
  - Default: 2 Conv1D layers, 16 filters, kernel_size=1
  - Window size: 2 timesteps

### Feature Types
- **ts_only**: Time series only (baseline)
- **weekend**: Time series + Saturday/Sunday dummies
- **business_hour**: Time series + Business hour dummy

## Training Strategy

All models use a **rolling window retraining approach**:
- Train on 3 years of data (26,280 hours)
- Forecast 24 hours ahead
- Retrain after each forecast period
- Update history with actual observed values

## Results Summary

Based on the original notebooks:

| Model | Features | RMSE |
|-------|----------|------|
| CNN | Business Hour | **1,359.49** ⭐ |
| CNN | Weekend | 2,392.55 |
| LSTM | Business Hour | 2,569.14 |
| LSTM | Weekend | 2,678.87 |
| MLP | Weekend | 2,716.58 |
| MLP | Business Hour | 2,938.88 |

**Best Model**: CNN with Business Hour features

## Improvements

✅ **Separation of Concerns**:
- Model definitions separated from training logic
- Easy to modify architectures without changing training code
- Reusable model builders

✅ **Fixed Bugs**:
- Corrected RMSE calculation (proper parentheses)
- Fixed scaler usage inconsistencies
- Removed hardcoded paths

✅ **Better Integration**:
- Uses `data_preprocessing.py` module
- No code duplication
- Consistent preprocessing

✅ **Enhanced Features**:
- Additional metrics (MAE, R²)
- Model saving capability
- Results export to CSV
- Better error handling
- Customizable model architectures

✅ **Code Quality**:
- Type hints
- Comprehensive documentation
- Modular design
- Configurable parameters

## Dependencies

```bash
pip install tensorflow scikit-learn pandas numpy
```

For Google Drive downloads:
```bash
pip install gdown
```

## Example Usage

See `example_usage.py` for detailed examples:
- Single model training
- Feature comparison
- Model comparison
- Running all configurations
- Custom model architectures

## Customization

### Custom Model Architecture

```python
from models import ModelBuilder

builder = ModelBuilder()

# Build custom MLP
model = builder.build_mlp(
    input_shape=(25,),
    hidden_units=50,
    n_layers=3,
    dropout_rate=0.3,
    use_batch_norm=True
)

# Compile with custom settings
builder.compile_model(
    model,
    learning_rate=0.001,
    loss='mae'
)
```

### Custom Training Parameters

```python
trainer = ModelTrainer(
    model_type='cnn',
    feature_type='business_hour',
    window_size=2,
    epochs=200,  # More epochs
    batch_size=256,  # Larger batch
    learning_rate=0.001,  # Lower learning rate
    patience=5,  # More patience
    # Custom model parameters
    filters=32,
    n_layers=3,
    dropout_rate=0.2
)
```

## Notes

- The original notebooks had some bugs that are fixed in `trainer.py`
- All models use the same rolling window training strategy
- Business hour feature consistently improves performance
- CNN performs best overall, especially with business hour features
- Model architectures can be easily customized via `models.py`
