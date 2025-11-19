"""
Comprehensive data preprocessing module for Italian energy data.

This module provides functions for:
- Downloading data from Google Drive
- Loading and combining multi-year CSV files
- Data cleaning and preprocessing
- Feature engineering (temporal features, aggregations)
- Missing value handling
"""

import os
import warnings
import pandas as pd
import numpy as np
from typing import List, Optional, Union
import requests
from pathlib import Path


def download_from_google_drive(
    folder_id: str,
    output_dir: str = "data",
    file_pattern: str = "ITA*.csv",
    use_gdown: bool = True
) -> List[str]:
    """
    Download CSV files from a Google Drive folder.
    
    This function supports two methods:
    1. Using gdown library (recommended, faster)
    2. Using requests library (fallback, requires direct file links)
    
    Parameters:
    -----------
    folder_id : str
        Google Drive folder ID (extracted from the folder URL)
        Example: "1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H"
    output_dir : str
        Directory where files will be saved (default: "data")
    file_pattern : str
        Pattern to match files (default: "ITA*.csv")
    use_gdown : bool
        If True, use gdown library. If False, use requests (requires file IDs)
    
    Returns:
    --------
    List[str]
        List of paths to downloaded files
    
    Example:
    --------
    >>> folder_id = "1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H"
    >>> files = download_from_google_drive(folder_id, output_dir="data")
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if use_gdown:
        try:
            import gdown
            
            # Download entire folder using gdown
            folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
            gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)
            
            # Find all matching files
            downloaded_files = []
            for file in Path(output_dir).rglob(file_pattern):
                downloaded_files.append(str(file))
            
            if not downloaded_files:
                warnings.warn(f"No files matching pattern '{file_pattern}' found in {output_dir}")
            
            return downloaded_files
            
        except ImportError:
            warnings.warn(
                "gdown library not found. Install it with: pip install gdown\n"
                "Falling back to manual download method."
            )
            use_gdown = False
    
    if not use_gdown:
        # Alternative: Manual download instructions
        print(f"\nTo download files manually:")
        print(f"1. Open the Google Drive folder: https://drive.google.com/drive/folders/{folder_id}")
        print(f"2. Download the CSV files matching '{file_pattern}'")
        print(f"3. Place them in the '{output_dir}' directory")
        print(f"\nOr install gdown: pip install gdown")
        
        # Check if files already exist
        existing_files = []
        if os.path.exists(output_dir):
            for file in Path(output_dir).glob(file_pattern):
                existing_files.append(str(file))
        
        return existing_files


def retrieve_data(
    path: Union[str, Path],
    years: Optional[List[int]] = None,
    handle_missing: bool = True
) -> pd.DataFrame:
    """
    Retrieve and combine energy data from multiple CSV files.
    
    Parameters:
    -----------
    path : str or Path
        Path to directory containing CSV files
    years : list of int, optional
        List of years to load. If None, defaults to [2016, 2017, 2018, 2019, 2020, 2021]
    handle_missing : bool
        If True, handle missing files gracefully. If False, raise error.
    
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with Time as index
    
    Example:
    --------
    >>> data = retrieve_data("data", years=[2019, 2020, 2021])
    """
    columns = [
        'Biomass  - Actual Aggregated [MW]',
        'Fossil Coal-derived gas  - Actual Aggregated [MW]',
        'Fossil Gas  - Actual Aggregated [MW]',
        'Fossil Hard coal  - Actual Aggregated [MW]',
        'Fossil Oil  - Actual Aggregated [MW]',
        'Geothermal  - Actual Aggregated [MW]',
        'Hydro Pumped Storage  - Actual Aggregated [MW]',
        'Hydro Run-of-river and poundage  - Actual Aggregated [MW]',
        'Hydro Water Reservoir  - Actual Aggregated [MW]',
        'Other  - Actual Aggregated [MW]',
        'Solar  - Actual Aggregated [MW]',
        'Waste  - Actual Aggregated [MW]',
        'Wind Onshore  - Actual Aggregated [MW]',
        'Time'
    ]

    # Default years if not specified
    if years is None:
        years = [2016, 2017, 2018, 2019, 2020, 2021]
    
    dataframes = []
    
    # Load data for each year
    for year in years:
        file_path = os.path.join(path, f"ITA{year}.csv")
        
        if not os.path.exists(file_path):
            if handle_missing:
                warnings.warn(f"File not found: {file_path}. Skipping year {year}.", UserWarning)
                continue
            else:
                raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            df = pd.read_csv(file_path, parse_dates=['MTU'])
            
            # Vectorized date formatting - truncate to first 16 characters (YYYY-MM-DD HH:MM)
            df['MTU'] = df['MTU'].astype(str).str[:16]
            
            # Convert to datetime
            df['Time'] = pd.to_datetime(df['MTU'], utc=True, format='%Y-%m-%d %H:%M', errors='coerce')
            
            # Format as string for consistency
            df['Time'] = df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['Time'])
            
            # Select and sort by Time
            available_columns = [col for col in columns if col in df.columns]
            df = df[available_columns].sort_values(by=['Time'])
            
            dataframes.append(df)
            
        except Exception as e:
            warnings.warn(f"Error loading {file_path}: {str(e)}. Skipping year {year}.", UserWarning)
            continue
    
    if not dataframes:
        raise ValueError("No data files found. Please check the path and years.")
    
    # Combine all dataframes
    tot = pd.concat(dataframes, ignore_index=True)
    
    # Set Time as index
    tot = tot.set_index(pd.DatetimeIndex(tot['Time']))
    tot = tot.sort_index()
    
    return tot


def handle_missing_values(df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Interpolation method: 'linear', 'forward', 'backward', 'polynomial', 'spline'
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values filled
    """
    df = df.copy()
    
    missing_count = df.isnull().values.sum()
    if missing_count > 0:
        print(f'Found {missing_count} missing values. Interpolating using {method} method...')
        df.interpolate(method=method, limit_direction='forward', inplace=True, axis=0)
        # Fill any remaining NaNs with forward fill, then backward fill
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
    else:
        print('No missing values found.')
    
    return df


def data_and_aggregator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate energy sources and create total columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with energy source columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with aggregated columns
    """
    df = df.copy()
    
    # Select all energy source columns (exclude Time and other non-energy columns)
    energy_cols = [col for col in df.columns 
                   if 'Actual Aggregated' in col or col in ['biomass', 'hard_coal', 'oil', 
                                                             'geothermal', 'other', 'solar', 
                                                             'waste', 'wind', 'hydro_tot', 'gas_tot']]
    
    # Create total aggregated column
    df['total_aggregated'] = df[energy_cols].sum(axis=1)
    
    # Aggregate hydroelectric sources
    hydro_cols = [
        'Hydro Pumped Storage  - Actual Aggregated [MW]',
        'Hydro Run-of-river and poundage  - Actual Aggregated [MW]',
        'Hydro Water Reservoir  - Actual Aggregated [MW]'
    ]
    
    # Only aggregate if columns exist
    existing_hydro_cols = [col for col in hydro_cols if col in df.columns]
    if existing_hydro_cols:
        df['hydro_tot'] = df[existing_hydro_cols].sum(axis=1)
        df = df.drop(columns=existing_hydro_cols)
    
    # Aggregate gas sources
    gas_cols = [
        'Fossil Coal-derived gas  - Actual Aggregated [MW]',
        'Fossil Gas  - Actual Aggregated [MW]'
    ]
    
    # Only aggregate if columns exist
    existing_gas_cols = [col for col in gas_cols if col in df.columns]
    if existing_gas_cols:
        df['gas_tot'] = df[existing_gas_cols].sum(axis=1)
        df = df.drop(columns=existing_gas_cols)
    
    # Rename columns to shorter names
    rename_dict = {
        'Fossil Hard coal  - Actual Aggregated [MW]': 'hard_coal',
        'Fossil Oil  - Actual Aggregated [MW]': 'oil',
        'Geothermal  - Actual Aggregated [MW]': 'geothermal',
        'Waste  - Actual Aggregated [MW]': 'waste',
        'Other  - Actual Aggregated [MW]': 'other',
        'Solar  - Actual Aggregated [MW]': 'solar',
        'Wind Onshore  - Actual Aggregated [MW]': 'wind',
        'Biomass  - Actual Aggregated [MW]': 'biomass'
    }
    
    # Only rename columns that exist
    rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    return df


def businesshour_and_we_generation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate temporal features: weekend, hour, saturday, sunday, and business hour.
    Uses vectorized operations for efficiency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Vectorized feature generation
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday  # 0=Monday, 6=Sunday
    
    # Weekend encoding: 0=weekday, 1=saturday, 2=sunday
    df['weekend'] = 0
    df.loc[df.index.weekday == 5, 'weekend'] = 1  # Saturday
    df.loc[df.index.weekday == 6, 'weekend'] = 2  # Sunday
    
    # Saturday and Sunday flags
    df['saturday'] = (df.index.weekday == 5).astype(int)
    df['sunday'] = (df.index.weekday == 6).astype(int)
    
    # Business hours: 8 AM to 6 PM (inclusive)
    df['business hour'] = ((df.index.hour >= 8) & (df.index.hour <= 18)).astype(int)
    
    return df


def preprocess_pipeline(
    data_path: Union[str, Path],
    years: Optional[List[int]] = None,
    download_from_drive: bool = False,
    drive_folder_id: Optional[str] = None,
    output_dir: str = "data",
    handle_missing_vals: bool = True,
    interpolate_method: str = 'linear'
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for energy data.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to directory containing CSV files, or where to save downloaded files
    years : list of int, optional
        List of years to process. If None, uses default [2016-2021]
    download_from_drive : bool
        If True, download files from Google Drive first
    drive_folder_id : str, optional
        Google Drive folder ID (required if download_from_drive=True)
    output_dir : str
        Directory for downloaded files (default: "data")
    handle_missing_vals : bool
        If True, interpolate missing values
    interpolate_method : str
        Interpolation method for missing values
    
    Returns:
    --------
    pd.DataFrame
        Fully preprocessed dataframe
    
    Example:
    --------
    >>> # Download and preprocess data
    >>> folder_id = "1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H"
    >>> data = preprocess_pipeline(
    ...     data_path="data",
    ...     download_from_drive=True,
    ...     drive_folder_id=folder_id,
    ...     years=[2019, 2020, 2021]
    ... )
    """
    # Step 1: Download from Google Drive if requested
    if download_from_drive:
        if drive_folder_id is None:
            raise ValueError("drive_folder_id must be provided when download_from_drive=True")
        
        print(f"Downloading files from Google Drive folder: {drive_folder_id}")
        downloaded_files = download_from_google_drive(
            folder_id=drive_folder_id,
            output_dir=output_dir,
            file_pattern="ITA*.csv"
        )
        
        if not downloaded_files:
            raise ValueError("No files were downloaded. Please check the folder ID and permissions.")
        
        data_path = output_dir
    
    # Step 2: Load data
    print(f"Loading data from: {data_path}")
    data = retrieve_data(data_path, years=years)
    
    # Step 3: Handle missing values
    if handle_missing_vals:
        data = handle_missing_values(data, method=interpolate_method)
    
    # Step 4: Aggregate energy sources
    print("Aggregating energy sources...")
    data = data_and_aggregator(data)
    
    # Step 5: Generate temporal features
    print("Generating temporal features...")
    data = businesshour_and_we_generation(data)
    
    # Step 6: Remove any remaining NaN values
    data = data.dropna()
    
    print(f"Preprocessing complete! Final shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    return data


if __name__ == "__main__":
    # Example usage
    folder_id = "1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H"
    
    # Option 1: Download and preprocess
    # data = preprocess_pipeline(
    #     data_path="data",
    #     download_from_drive=True,
    #     drive_folder_id=folder_id,
    #     years=[2019, 2020, 2021]
    # )
    
    # Option 2: Use existing local files
    # data = preprocess_pipeline(
    #     data_path="data",
    #     years=[2019, 2020, 2021]
    # )
    
    print("Data preprocessing module loaded successfully!")
    print("Use preprocess_pipeline() to process your data.")

