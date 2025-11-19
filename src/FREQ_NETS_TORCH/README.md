# FREQ_NETS_TORCH - Frequency Domain Neural Networks (PyTorch)

This folder contains PyTorch implementations of MLP, LSTM, and CNN models for time series forecasting of Italian energy generation data.

This is the PyTorch equivalent of the `FREQ_NETS_TF` folder, providing the same functionality using PyTorch instead of TensorFlow.

## File Structure

### Core Modules
- **`models.py`**: PyTorch model architecture definitions (MLP, LSTM, CNN)
- **`trainer.py`**: Training and testing logic with rolling window approach
- **`Unified_Training_Notebook.ipynb`**: Unified notebook for all PyTorch models

### Documentation
- **`README.md`**: This file
- **`example_usage.py`**: Example scripts showing different usage patterns

## Architecture

The code is split into two main modules:

### 1. `models.py` - Model Building
Contains PyTorch model architecture definitions:
- `MLP`: Multi-Layer Perceptron (nn.Module)
- `LSTM`: Long Short-Term Memory (nn.Module)
- `CNN`: Convolutional Neural Network (nn.Module)
- `ModelBuilder`: Class with static methods to build models
- `create_model()`: Factory function to create models
- `DEFAULT_CONFIGS`: Default configurations matching TensorFlow versions

### 2. `trainer.py` - Training & Testing
Contains PyTorch training and evaluation logic:
- `DataPreparator`: Handles data preparation for different model types
- `ModelTrainer`: Main training class with rolling window approach
- `EarlyStopping`: Custom early stopping implementation
- `run_all_models()`: Function to run all model configurations

## Key Differences from TensorFlow Version

1. **Custom Training Loop**: PyTorch uses explicit training loops instead of `model.fit()`
2. **Tensors**: Uses PyTorch tensors instead of numpy arrays
3. **DataLoader**: Uses PyTorch DataLoader for batching
4. **Device Management**: Explicit CPU/GPU device handling
5. **Model Saving**: Saves model state dict instead of full model

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
    verbose=True,
    device='cuda'  # or 'cpu'
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
    n_train=35064,
    device='cuda'  # Auto-detect if None
)
```

### Option 3: Command Line

```bash
# Run all models
python trainer.py --model all --features all

# Run specific model
python trainer.py --model cnn --features business_hour --device cuda

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

## Device Support

PyTorch automatically detects and uses GPU if available:
- **CPU**: Default fallback
- **CUDA**: Automatically used if available
- **Manual**: Specify `device='cuda'` or `device='cpu'`

## Dependencies

```bash
pip install torch scikit-learn pandas numpy matplotlib
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
    input_size=25,
    hidden_units=50,
    n_layers=3,
    dropout_rate=0.3,
    use_batch_norm=True
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
    device='cuda',  # Use GPU
    # Custom model parameters
    filters=32,
    n_layers=3,
    dropout_rate=0.2
)
```

## Comparison with TensorFlow Version

| Feature | TensorFlow | PyTorch |
|---------|------------|---------|
| Model Definition | Sequential API | nn.Module classes |
| Training | `model.fit()` | Custom training loop |
| Data Handling | Numpy arrays | PyTorch tensors |
| Batching | Built-in | DataLoader |
| Device Management | Automatic | Explicit |
| Model Saving | `.h5` format | `.pth` state dict |

Both versions produce equivalent results and use the same:
- Data preprocessing pipeline
- Rolling window training strategy
- Evaluation metrics
- Feature engineering

## Building Models with PyTorch

### Understanding PyTorch Models

PyTorch models are defined as classes that inherit from `nn.Module`. Each model must implement:
1. **`__init__()`**: Define the layers and components
2. **`forward()`**: Define the forward pass (how data flows through the model)

### Model Architecture Examples

#### MLP (Multi-Layer Perceptron)

```python
from models import MLP

# Create an MLP model
model = MLP(
    input_size=25,        # Number of input features (e.g., 24 lags + 1 dummy)
    hidden_units=10,      # Neurons in each hidden layer
    n_layers=2,           # Number of hidden layers
    activation='relu',    # Activation function
    dropout_rate=0.2,     # Dropout for regularization (optional)
    use_batch_norm=False  # Batch normalization (optional)
)

# Move model to device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Print model architecture
print(model)
```

**Architecture Structure:**
```
Input (25 features)
  ↓
Linear(25 → 10) + ReLU + Dropout(0.2)
  ↓
Linear(10 → 10) + ReLU + Dropout(0.2)
  ↓
Linear(10 → 1)  (Output)
```

#### LSTM (Long Short-Term Memory)

```python
from models import LSTM

# Create an LSTM model
model = LSTM(
    input_size=3,         # Features per timestep (e.g., total_aggregated + saturday + sunday)
    lstm_units=10,        # Hidden size of LSTM
    n_layers=2,           # Number of stacked LSTM layers
    dropout_rate=0.2,     # Dropout between LSTM layers
    return_sequences=True # Return all timesteps (for stacked LSTMs)
)

model = model.to(device)
```

**Architecture Structure:**
```
Input (batch, seq_len=1, features=3)
  ↓
LSTM(3 → 10, num_layers=2, dropout=0.2)
  ↓
Flatten
  ↓
Linear(10 → 1)  (Output)
```

#### CNN (Convolutional Neural Network)

```python
from models import CNN

# Create a CNN model
model = CNN(
    input_size=2,         # Features per timestep
    filters=16,          # Number of convolutional filters
    kernel_size=1,       # Size of convolution kernel
    n_layers=2,          # Number of convolutional layers
    activation='relu',
    dropout_rate=0.2,
    use_pooling=False    # Whether to use max pooling
)

model = model.to(device)
```

**Architecture Structure:**
```
Input (batch, seq_len=2, features=2)
  ↓
Transpose to (batch, features=2, seq_len=2)
  ↓
Conv1d(2 → 16, kernel=1) + ReLU + Dropout
  ↓
Conv1d(16 → 16, kernel=1) + ReLU + Dropout
  ↓
AdaptiveAvgPool1d(1)  (Global pooling)
  ↓
Linear(16 → 1)  (Output)
```

### Using the ModelBuilder

The `ModelBuilder` class provides convenient static methods:

```python
from models import ModelBuilder

builder = ModelBuilder()

# Build models using builder
mlp = builder.build_mlp(input_size=25, hidden_units=20, n_layers=3)
lstm = builder.build_lstm(input_size=3, lstm_units=20, n_layers=2)
cnn = builder.build_cnn(input_size=2, filters=32, n_layers=3)
```

### Using the Factory Function

```python
from models import create_model

# Simple model creation
model = create_model('mlp', input_size=25)

# Custom model with parameters
model = create_model(
    'cnn',
    input_size=2,
    filters=32,
    n_layers=3,
    dropout_rate=0.2
)
```

## Training Procedure

### Step-by-Step Training Process

The training procedure follows these steps:

#### 1. **Data Preparation**

```python
from trainer import DataPreparator

# Initialize data preparator
preparator = DataPreparator(
    model_type='mlp',
    feature_type='business_hour',
    window_size=24
)

# Prepare features from DataFrame
feature_array, feature_names = preparator.prepare_features(data)

# Split into train/test
train_data_raw = feature_array[:n_train]
test_data_raw = feature_array[n_train:]

# Scale features (only scales first column, leaves dummies unchanged)
train_data = preparator.scale_features(train_data_raw, fit=True)
test_data = preparator.scale_features(test_data_raw, fit=False)

# Prepare data for model (creates lags/windows)
X_train, y_train, X_test, y_test = preparator.prepare_data_for_model(
    train_data, test_data
)
```

#### 2. **Model Initialization**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models import create_model

# Determine input size based on model type
if model_type == 'mlp':
    input_size = X_train.shape[1]  # Number of features
elif model_type in ['lstm', 'cnn']:
    input_size = X_train.shape[2]  # Features per timestep

# Create model
model = create_model(model_type, input_size, **config)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Initialize optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
```

#### 3. **Create DataLoader**

```python
from torch.utils.data import DataLoader, TensorDataset

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.FloatTensor(X_train)
y_tensor = torch.FloatTensor(y_train)

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True  # Shuffle for better training
)
```

#### 4. **Training Loop (One Epoch)**

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()  # Set model to training mode
    total_loss = 0.0
    n_batches = 0
    
    for batch_X, batch_y in dataloader:
        # Move data to device (CPU or GPU)
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Zero gradients (important!)
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(batch_X)
        
        # Compute loss
        loss = criterion(outputs.squeeze(), batch_y)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches
```

#### 5. **Complete Training with Early Stopping**

```python
from trainer import EarlyStopping

# Initialize early stopping
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

# Training loop
for epoch in range(max_epochs):
    # Train one epoch
    train_loss = train_epoch(model, dataloader, criterion, optimizer, device)
    
    # Check early stopping
    if early_stopping(train_loss, model):
        print(f"Early stopping at epoch {epoch + 1}")
        break
    
    if verbose and (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {train_loss:.4f}")
```

#### 6. **Making Predictions**

```python
def predict(model, X, device):
    """Make predictions."""
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for efficiency
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)
        return predictions.cpu().numpy()  # Move back to CPU and convert to numpy

# Make predictions
yhat = predict(model, X_test, device)

# Inverse transform (if data was scaled)
yhat_rescaled = scaler.inverse_transform(yhat.reshape(-1, 1)).flatten()
```

### Rolling Window Training Strategy

The training uses a **rolling window approach** for time series forecasting:

```python
# Initialize history with training data
history_X = X_train.copy()
history_y = y_train.copy()

# For each forecast period
for i in range(0, len(X_test), n_forecast_steps):
    # 1. Train model on current history
    model = create_model(...)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        train_epoch(model, dataloader, criterion, optimizer, device)
    
    # 2. Predict next period
    X_test_block = X_test[i:i+n_forecast_steps]
    yhat = predict(model, X_test_block, device)
    
    # 3. Update history with actual observed values
    history_X = np.concatenate((history_X, X_test_block), axis=0)
    history_X = history_X[n_forecast_steps:]  # Remove oldest data
    
    history_y = np.concatenate((history_y, y_test[i:i+n_forecast_steps]), axis=0)
    history_y = history_y[n_forecast_steps:]  # Remove oldest data
```

**Why Rolling Window?**
- Adapts to changing patterns over time
- Uses most recent data for training
- Maintains fixed training window size
- More realistic for production forecasting

## Testing and Evaluation

### Evaluation Metrics

The training procedure automatically computes:

1. **RMSE (Root Mean Squared Error)**
   ```python
   rmse = sqrt(mean_squared_error(y_true, y_pred))
   ```

2. **MAE (Mean Absolute Error)**
   ```python
   mae = mean_absolute_error(y_true, y_pred)
   ```

3. **R² (Coefficient of Determination)**
   ```python
   r2 = r2_score(y_true, y_pred)
   ```

### Manual Evaluation

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# Make predictions
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    predictions = model(X_test_tensor).cpu().numpy()

# Inverse transform
predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
truth = test_data_raw[-len(predictions):, 0]

# Calculate metrics
rmse = sqrt(mean_squared_error(truth, predictions_rescaled))
mae = mean_absolute_error(truth, predictions_rescaled)
r2 = r2_score(truth, predictions_rescaled)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
```

### Model Saving and Loading

#### Saving a Model

```python
# Save model state dict (recommended)
torch.save({
    'model_state_dict': model.state_dict(),
    'model_type': 'cnn',
    'input_size': 2,
    'config': {'filters': 16, 'n_layers': 2},
    'scaler': scaler  # Save scaler for inverse transform
}, 'model.pth')
```

#### Loading a Model

```python
from models import create_model

# Load checkpoint
checkpoint = torch.load('model.pth', map_location=device)

# Recreate model architecture
model = create_model(
    checkpoint['model_type'],
    checkpoint['input_size'],
    **checkpoint['config']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()  # Set to evaluation mode

# Load scaler if saved
scaler = checkpoint.get('scaler', None)
```

## Complete Training Example

Here's a complete example of training a model from scratch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import create_model
from trainer import DataPreparator, EarlyStopping
from data_preprocessing import preprocess_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# 1. Load and preprocess data
data = preprocess_pipeline(data_path="data", download_from_drive=False)

# 2. Prepare data
preparator = DataPreparator('cnn', 'business_hour', window_size=2)
feature_array, _ = preparator.prepare_features(data)

n_train = 35064
train_data_raw = feature_array[:n_train]
test_data_raw = feature_array[n_train:]

train_data = preparator.scale_features(train_data_raw, fit=True)
test_data = preparator.scale_features(test_data_raw, fit=False)

X_train, y_train, X_test, y_test = preparator.prepare_data_for_model(
    train_data, test_data
)

# 3. Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 4. Create model
input_size = X_train.shape[2]  # Features per timestep for CNN
model = create_model('cnn', input_size, filters=16, n_layers=2)
model = model.to(device)

# 5. Setup training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
early_stopping = EarlyStopping(patience=3)

# 6. Create DataLoader
X_tensor = torch.FloatTensor(X_train).to(device)
y_tensor = torch.FloatTensor(y_train).to(device)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=168, shuffle=True)

# 7. Training loop
for epoch in range(100):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    
    # Early stopping check
    if early_stopping(avg_loss, model):
        print(f"Early stopping at epoch {epoch + 1}")
        break
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# 8. Evaluation
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    predictions = model(X_test_tensor).cpu().numpy()

predictions_rescaled = preparator.scaler.inverse_transform(
    predictions.reshape(-1, 1)
).flatten()
truth = test_data_raw[-len(predictions):, 0]

rmse = sqrt(mean_squared_error(truth, predictions_rescaled))
mae = mean_absolute_error(truth, predictions_rescaled)
r2 = r2_score(truth, predictions_rescaled)

print(f"\nFinal Results:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
```

## Key PyTorch Concepts

### 1. **Tensors**
- PyTorch uses tensors (similar to numpy arrays) but with GPU support
- Convert numpy to tensor: `torch.FloatTensor(array)`
- Convert tensor to numpy: `tensor.cpu().numpy()`

### 2. **Device Management**
- Always move model and data to the same device
- `model.to(device)` and `data.to(device)`
- Use `torch.cuda.is_available()` to check GPU availability

### 3. **Training vs Evaluation Mode**
- `model.train()`: Enables dropout, batch norm updates
- `model.eval()`: Disables dropout, uses fixed batch norm stats
- Always use `model.eval()` when making predictions

### 4. **Gradient Computation**
- `torch.no_grad()`: Disables gradient computation (faster, less memory)
- Use during inference/evaluation
- Always use `optimizer.zero_grad()` before backward pass

### 5. **Loss Functions**
- `nn.MSELoss()`: Mean Squared Error (default for regression)
- `nn.L1Loss()`: Mean Absolute Error
- Loss is a tensor, use `.item()` to get Python float

### 6. **Optimizers**
- `optim.Adam()`: Adaptive learning rate (recommended)
- `optim.SGD()`: Stochastic Gradient Descent
- Always call `optimizer.step()` after `loss.backward()`

## Best Practices

1. **Always use `model.eval()` for inference**
   ```python
   model.eval()
   with torch.no_grad():
       predictions = model(X_test)
   ```

2. **Move data to device efficiently**
   ```python
   # Good: Move once
   X_tensor = torch.FloatTensor(X).to(device)
   
   # Bad: Moving in loop
   for x in X:
       x_tensor = torch.FloatTensor(x).to(device)  # Slow!
   ```

3. **Use DataLoader for batching**
   ```python
   # Efficient batching and shuffling
   dataloader = DataLoader(dataset, batch_size=168, shuffle=True)
   ```

4. **Save model state dict, not full model**
   ```python
   # Good: Saves only weights
   torch.save(model.state_dict(), 'model.pth')
   
   # Less flexible: Saves entire model
   torch.save(model, 'model.pth')
   ```

5. **Use early stopping to prevent overfitting**
   ```python
   early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
   ```

6. **Monitor training with loss values**
   ```python
   # Track loss per epoch
   train_losses = []
   for epoch in range(epochs):
       loss = train_epoch(...)
       train_losses.append(loss)
   ```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller model
   - Use gradient accumulation

2. **Model Not Learning**
   - Check learning rate (try 0.001, 0.0001)
   - Check data scaling
   - Check if gradients are flowing: `print(model.parameters())`

3. **Slow Training**
   - Use GPU: `device='cuda'`
   - Increase batch size (if memory allows)
   - Use mixed precision training

4. **Inconsistent Results**
   - Set random seeds:
     ```python
     torch.manual_seed(42)
     np.random.seed(42)
     ```

## Notes

- Models are equivalent to TensorFlow versions in architecture
- Same rolling window training strategy
- Business hour feature consistently improves performance
- CNN performs best overall, especially with business hour features
- GPU acceleration available for faster training
- Model architectures can be easily customized via `models.py`
- PyTorch provides more flexibility and control over training process

