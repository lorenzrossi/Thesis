# Italian Energy Generation Forecasting

A comprehensive machine learning project for forecasting Italian energy generation using neural networks. This project includes implementations in both TensorFlow and PyTorch, with support for frequentist and Bayesian approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Documentation](#documentation)
- [Contributing](#contributing)

## 🎯 Overview

This project implements time series forecasting models for Italian energy generation data (2016-2021). The models use a rolling window training approach to predict future energy generation based on historical patterns.

### Key Components

- **Data Preprocessing**: Automated pipeline for downloading, cleaning, and feature engineering
- **Frequentist Models**: MLP, LSTM, and CNN implementations in TensorFlow and PyTorch
- **Bayesian Models**: Bayesian neural network implementations (PyTorch)
- **Unified Training**: Single notebooks for training all model configurations

## 📁 Project Structure

```
Thesis/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
└── src/                     # Source code
    ├── data_preprocessing.py      # Main preprocessing module
    ├── preprocessing_utils.py     # Original utility functions
    ├── example_preprocessing.py   # Preprocessing examples
    ├── DATA_DESC.ipynb            # Exploratory data analysis
    │
    ├── FREQ_NETS_TF/              # TensorFlow implementations
    │   ├── models.py              # Model architectures
    │   ├── trainer.py             # Training logic
    │   ├── Unified_Training_Notebook.ipynb
    │   ├── README.md
    │   └── ...
    │
    ├── FREQ_NETS_TORCH/           # PyTorch implementations
    │   ├── models.py              # Model architectures
    │   ├── trainer.py             # Training logic
    │   ├── Unified_Training_Notebook.ipynb
    │   └── README.md
    │
    └── BAYES_NOTEBOOKS_PYTORCH/   # Bayesian models
        └── ...
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Thesis
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")  # For Apple Silicon
```

## 🏃 Quick Start

### 1. Preprocess Data

```python
from src.data_preprocessing import preprocess_pipeline

# Download and preprocess data
data = preprocess_pipeline(
    data_path="data",
    download_from_drive=True,
    drive_folder_id="1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H",
    years=[2019, 2020, 2021]
)
```

### 2. Train a Model (PyTorch)

```python
from src.FREQ_NETS_TORCH.trainer import ModelTrainer

trainer = ModelTrainer(
    model_type='mlp',
    feature_type='ts_only',
    window_size=24,
    n_forecast_steps=24,
    epochs=100,
    batch_size=168,
    learning_rate=0.003,
    device='mps'  # or 'cpu', 'cuda'
)

results = trainer.train(data, n_train=35064)
print(f"RMSE: {results['overall_rmse']:.2f}")
```

### 3. Use Jupyter Notebooks

```bash
jupyter notebook src/FREQ_NETS_TORCH/Unified_Training_Notebook.ipynb
```

## ✨ Features

### Data Preprocessing
- ✅ Automatic data download from Google Drive
- ✅ Missing value handling with interpolation
- ✅ Energy source aggregation
- ✅ Temporal feature engineering (weekend, business hours)
- ✅ Vectorized operations for efficiency

### Model Implementations
- **MLP**: Multi-Layer Perceptron for time series forecasting
- **LSTM**: Long Short-Term Memory networks
- **CNN**: 1D Convolutional Neural Networks
- **Bayesian**: Bayesian neural network variants

### Training Features
- Rolling window training approach
- Early stopping
- Multiple feature combinations (time series only, weekend dummies, business hour)
- Comprehensive evaluation metrics (RMSE, MAE, R²)
- GPU support (CUDA, MPS for Apple Silicon)

## 📚 Documentation

- **[Data Preprocessing](src/README.md)**: Detailed preprocessing documentation
- **[TensorFlow Models](src/FREQ_NETS_TF/README.md)**: TensorFlow implementation guide
- **[PyTorch Models](src/FREQ_NETS_TORCH/README.md)**: PyTorch implementation guide with detailed examples

## 🔧 Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 2.0.0 (with MPS support for macOS)
- TensorFlow >= 2.0.0
- NumPy, Pandas
- scikit-learn
- Matplotlib
- Jupyter

## 📊 Data

The project uses Italian energy generation data from 2016-2021, available via Google Drive. The preprocessing pipeline handles:
- Multiple energy sources (biomass, coal, gas, hydro, solar, wind, etc.)
- Temporal features (hour, weekday, weekend, business hours)
- Aggregated totals

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- Italian energy data sources
- PyTorch and TensorFlow communities

---

For detailed usage examples, see the README files in each subdirectory.
