# Data Preprocessing Module

This module provides comprehensive data preprocessing tools for Italian energy generation data (2016-2021).

### Overview
This project includes comprehensive data preprocessing tools for Italian energy generation data (2016-2021). The preprocessing pipeline handles data loading, cleaning, aggregation, and feature engineering.

### Quick Start

#### Option 1: Download and Preprocess (Recommended)
```python
from data_preprocessing import preprocess_pipeline

# Download from Google Drive and preprocess
folder_id = "1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H"
data = preprocess_pipeline(
    data_path="data",
    download_from_drive=True,
    drive_folder_id=folder_id,
    years=[2019, 2020, 2021]
)
```

#### Option 2: Use Existing Local Files
```python
from data_preprocessing import preprocess_pipeline

# Use files already in your data directory
data = preprocess_pipeline(
    data_path="data",
    download_from_drive=False,
    years=[2019, 2020, 2021]
)
```

### Installation

For Google Drive downloads, install `gdown`:
```bash
pip install gdown
```

### Files

- **`data_preprocessing.py`**: Comprehensive preprocessing module with Google Drive download support
- **`preprocessing_utils.py`**: Original utility functions (updated with bug fixes)
- **`example_preprocessing.py`**: Example scripts showing different usage patterns
- **`DATA_DESC.ipynb`**: Jupyter notebook for exploratory data analysis

### Features

1. **Flexible Data Download**: Download CSV files from Google Drive automatically
2. **Vectorized Operations**: Efficient preprocessing using pandas vectorization
3. **Missing Value Handling**: Automatic interpolation and filling
4. **Energy Source Aggregation**: Combines related energy sources (hydro, gas)
5. **Temporal Features**: Generates weekend, business hour, and other time-based features
6. **Error Handling**: Graceful handling of missing files and data issues

### Improvements Made

- ✅ Fixed `SettingWithCopyWarning` by vectorizing date formatting
- ✅ Vectorized feature generation for better performance
- ✅ Added flexible Google Drive download functionality
- ✅ Improved error handling and validation
- ✅ Fixed bug in `data_and_aggregator` function
- ✅ Added comprehensive documentation and type hints

### Data Structure

The preprocessed data includes:
- **Energy Sources**: biomass, hard_coal, oil, geothermal, other, solar, waste, wind
- **Aggregated Sources**: hydro_tot, gas_tot, total_aggregated
- **Temporal Features**: hour, weekday, weekend, saturday, sunday, business hour
- **Index**: DatetimeIndex with hourly frequency

### Example Usage

See `example_preprocessing.py` for detailed examples of different usage patterns.
