import pandas as pd
from pathlib import Path


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    load play-by-play data from CSV file.
    
    Args:
        filepath: Optional path to CSV file. 
        
    Returns:
        dataFrame containing play-by-play data
    """
    if filepath is None:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        DATA_DIR = PROJECT_ROOT / "data"
        filepath = DATA_DIR / "raw_pbp_21_25.csv"
        if not filepath.exists():
            filepath = DATA_DIR / "raw_pbp_19_25.csv"
    
    try:
        df = pd.read_csv(filepath, low_memory=False) # help reduce errors
        print(f"Data loaded successfully from {filepath}")
        print(f"Shape: {df.shape}")
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    except Exception as e:
        raise Exception(f"Error loading data: {e}")