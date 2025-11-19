"""
Example script demonstrating how to use the data preprocessing module.

This script shows different ways to download and preprocess Italian energy data.
"""

from data_preprocessing import (
    download_from_google_drive,
    preprocess_pipeline,
    retrieve_data,
    handle_missing_values,
    data_and_aggregator,
    businesshour_and_we_generation
)

# Google Drive folder ID
FOLDER_ID = "1fgXJNVg3MUu8Vx8kAthW4dAih9rKme2H"


def example_1_download_and_preprocess():
    """
    Example 1: Download from Google Drive and preprocess in one step.
    """
    print("=" * 60)
    print("Example 1: Download and Preprocess Pipeline")
    print("=" * 60)
    
    data = preprocess_pipeline(
        data_path="data",
        download_from_drive=True,
        drive_folder_id=FOLDER_ID,
        output_dir="data",
        years=[2019, 2020, 2021],  # Process specific years
        handle_missing_vals=True,
        interpolate_method='linear'
    )
    
    print(f"\nData shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    return data


def example_2_use_existing_files():
    """
    Example 2: Use existing local CSV files.
    """
    print("\n" + "=" * 60)
    print("Example 2: Preprocess Existing Local Files")
    print("=" * 60)
    
    # Assuming files are already in the 'data' directory
    data = preprocess_pipeline(
        data_path="data",
        download_from_drive=False,
        years=[2020, 2021],
        handle_missing_vals=True
    )
    
    return data


def example_3_step_by_step():
    """
    Example 3: Step-by-step preprocessing for more control.
    """
    print("\n" + "=" * 60)
    print("Example 3: Step-by-Step Preprocessing")
    print("=" * 60)
    
    # Step 1: Download files (if needed)
    print("\nStep 1: Downloading files...")
    downloaded_files = download_from_google_drive(
        folder_id=FOLDER_ID,
        output_dir="data",
        file_pattern="ITA*.csv"
    )
    print(f"Downloaded {len(downloaded_files)} files")
    
    # Step 2: Load data
    print("\nStep 2: Loading data...")
    data = retrieve_data("data", years=[2021])
    
    # Step 3: Handle missing values
    print("\nStep 3: Handling missing values...")
    data = handle_missing_values(data, method='linear')
    
    # Step 4: Aggregate energy sources
    print("\nStep 4: Aggregating energy sources...")
    data = data_and_aggregator(data)
    
    # Step 5: Generate temporal features
    print("\nStep 5: Generating temporal features...")
    data = businesshour_and_we_generation(data)
    
    # Step 6: Final cleanup
    data = data.dropna()
    
    print(f"\nFinal data shape: {data.shape}")
    return data


def example_4_custom_path():
    """
    Example 4: Use a custom path for data files.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Data Path")
    print("=" * 60)
    
    # Use a custom path
    custom_path = "/path/to/your/data"
    
    data = preprocess_pipeline(
        data_path=custom_path,
        download_from_drive=False,
        years=[2016, 2017, 2018, 2019, 2020, 2021]
    )
    
    return data


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # Example 1: Full pipeline with download
    # data = example_1_download_and_preprocess()
    
    # Example 2: Use existing files
    # data = example_2_use_existing_files()
    
    # Example 3: Step-by-step
    # data = example_3_step_by_step()
    
    # Example 4: Custom path
    # data = example_4_custom_path()
    
    print("\n" + "=" * 60)
    print("Examples ready! Uncomment the example you want to run.")
    print("=" * 60)
    print("\nTo install gdown (for Google Drive downloads):")
    print("  pip install gdown")
    print("\nFor more information, see data_preprocessing.py")

