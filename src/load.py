import pandas as pd
from pathlib import Path
from typing import Optional


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load play-by-play data from CSV file.
    
    Args:
        filepath: Optional path to CSV file. If None, uses default data file.
        
    Returns:
        DataFrame containing play-by-play data
    """
    if filepath is None:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        DATA_DIR = PROJECT_ROOT / "data"
        # Try the file used in the notebook first, fallback to other file
        filepath = DATA_DIR / "raw_pbp_21_25.csv"
        if not filepath.exists():
            filepath = DATA_DIR / "raw_pbp_19_25.csv"
    
    try:
        # Use low_memory=False to avoid dtype warnings
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Data loaded successfully from {filepath}")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")