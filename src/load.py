import nflreadpy as nfl
import pandas as pd

def load_data() -> pd.DataFrame:
    pbp = nfl.load_pbp(seasons=[2021,2022,2023,2024,2025])
    df = pbp.to_pandas()

    df.to_csv('data/raw_pbp.csv')