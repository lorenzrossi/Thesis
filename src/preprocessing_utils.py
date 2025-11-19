import datetime
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings

def retrieve_data(path, years=None):
    """
    Retrieve and combine energy data from multiple CSV files.
    
    Parameters:
    -----------
    path : str
        Path to directory containing CSV files or path to a single CSV file
    years : list, optional
        List of years to load. If None, defaults to [2016, 2017, 2018, 2019, 2020, 2021]
    
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with Time as index
    """
    columns = ['Biomass  - Actual Aggregated [MW]',
       'Fossil Coal-derived gas  - Actual Aggregated [MW]',
       'Fossil Gas  - Actual Aggregated [MW]',
       'Fossil Hard coal  - Actual Aggregated [MW]',
       'Fossil Oil  - Actual Aggregated [MW]',
       'Geothermal  - Actual Aggregated [MW]',
       'Hydro Pumped Storage  - Actual Aggregated [MW]',
       'Hydro Run-of-river and poundage  - Actual Aggregated [MW]',
       'Hydro Water Reservoir  - Actual Aggregated [MW]',
       'Other  - Actual Aggregated [MW]', 'Solar  - Actual Aggregated [MW]',
       'Waste  - Actual Aggregated [MW]',
       'Wind Onshore  - Actual Aggregated [MW]', 'Time']

    # Default years if not specified
    if years is None:
        years = [2016, 2017, 2018, 2019, 2020, 2021]
    
    dataframes = []
    
    # Load data for each year
    for year in years:
        file_path = os.path.join(path, f"ITA{year}.csv")
        
        if not os.path.exists(file_path):
            warnings.warn(f"File not found: {file_path}. Skipping year {year}.", UserWarning)
            continue
            
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
        df = df[columns].sort_values(by=['Time'])
        
        dataframes.append(df)
    
    if not dataframes:
        raise ValueError("No data files found. Please check the path and years.")
    
    # Combine all dataframes
    tot = pd.concat(dataframes, ignore_index=True)
    
    # Set Time as index
    tot = tot.set_index(pd.DatetimeIndex(tot['Time']))
    tot = tot.sort_index()
    
    return tot

def data_and_aggregator(df):

    # seleziono tutte le colonne delle fonti energetiche
    columns = df.columns[ : df.shape[1]-1]

    # aggrego giornalmente i dati
    #daily_df = pd.DataFrame(df[columns].groupby([pd.Grouper(level='Time', freq='D')]).sum())

    # creo l'aggregato totale
    df['total_aggregated'] = df[columns].sum(axis=1)

    # aggrego i dati dell'idroelettrico
    hydro_cols = ['Hydro Pumped Storage  - Actual Aggregated [MW]',
       'Hydro Run-of-river and poundage  - Actual Aggregated [MW]',
       'Hydro Water Reservoir  - Actual Aggregated [MW]']
    
    df['hydro_tot'] = df[hydro_cols].sum(axis=1)

    # aggrego i dati del gas
    gas_cols = ['Fossil Coal-derived gas  - Actual Aggregated [MW]',
                'Fossil Gas  - Actual Aggregated [MW]']
    
    df['gas_tot'] = df[gas_cols].sum(axis=1)

    df = df.rename(columns={'Fossil Hard coal  - Actual Aggregated [MW]': 'hard_coal', 
                            'Fossil Oil  - Actual Aggregated [MW]': 'oil',
                            'Geothermal  - Actual Aggregated [MW]': 'geothermal',
                            'Waste  - Actual Aggregated [MW]': 'waste',
                            'Other  - Actual Aggregated [MW]': 'other',
                            'Solar  - Actual Aggregated [MW]' : 'solar',
                            'Wind Onshore  - Actual Aggregated [MW]' : 'wind',
                            'Biomass  - Actual Aggregated [MW]' : 'biomass'})
    
    # Drop aggregated columns (only if they exist)
    existing_hydro_cols = [col for col in hydro_cols if col in df.columns]
    existing_gas_cols = [col for col in gas_cols if col in df.columns]
    
    if existing_hydro_cols:
        df = df.drop(columns=existing_hydro_cols)
    if existing_gas_cols:
        df = df.drop(columns=existing_gas_cols)

    return df

def businesshour_and_we_generation(df):
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
    # Make a copy to avoid SettingWithCopyWarning
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