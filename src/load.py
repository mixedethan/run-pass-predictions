import nflreadpy as nfl
import pandas as pd
from pathlib import Path

def load_data() -> pd.DataFrame:
    pbp = nfl.load_pbp(seasons=[2019,2020,2021,2022,2023,2024,2025]) # 2021 - 2025 originally
    df = pbp.to_pandas()

    # /run-pass-predictions/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # /run-pass-predictions/data
    DATA_DIR = PROJECT_ROOT / "data"
    
    # make the directory unless it already exists
    DATA_DIR.mkdir(exist_ok=True)

    # /run-pass-predictions/data/raw_pbp_19_25.csv
    file_path = DATA_DIR / "raw_pbp_19_25.csv"

    df.to_csv(file_path, index=False)

    return df