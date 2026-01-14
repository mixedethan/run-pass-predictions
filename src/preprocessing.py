import pandas as pd
from config import * 

# 1) select our features
def select_features(df: pd.DataFrame) -> pd.DataFrame:
   
    cols_to_keep = [
        "yardline_100",  # numeric distance in the number of yards from the opponent's endzone for the posteam.
        "posteam",  # string abbreviation for the team with possession.
        "quarter_seconds_remaining",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "game_half",  # string indicating which half the play is in, either Half1, Half2, or Overtime.
        "drive",  # numeric drive number in the game.
        "qtr",  # quarter of the game (5 is overtime).
        "down",
        "goal_to_go",  # binary indicator for whether or not the posteam is in a goal down situation.
        "time",  # time at start of play provided in string format as minutes:seconds remaining in the quarter.
        "ydstogo",
        "play_type",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "posteam_score",
        "defteam_score",
        "score_differential",  # score differential between the posteam and defteam at the start of the play.
        "ep",  # estimated expected points with respect to the possession team for the given play.
        "posteam_type",  # string indicating whether the posteam team is home or away
        "wp",  # estimated win probability for the posteam given the current situation at the start of the given play
        "season"
    ]
    
    # only keep columns that exist in the dataframe
    available_cols = [col for col in cols_to_keep if col in df.columns]
    df_selected = df[available_cols].copy()
    
    return df_selected

# 2) select the needed play_types for the model
def filter_play_types(df: pd.DataFrame) -> pd.DataFrame:
   
    df_filtered = df[df['play_type'].isin(VALID_PLAY_TYPES)].copy()
    print(f"Filtered to {len(df_filtered)} plays (from {len(df)} total)")
    return df_filtered

# 3) clean the data by removing nulls, duplicates, and optionally garbage time.
def clean_data(df: pd.DataFrame, garbage_time_method: int = 2) -> pd.DataFrame:
    """
    clean the data by removing nulls, duplicates, and optionally garbage time.
    
    args:
        df: input dataframe to clean
        garbage_time_method: Method for garbage time removal (0, 1, or 2)
        
    Returns:
        cleaned dataframe
    """
    initial_len = len(df)
    
    # drop null rows
    df_cleaned = df.dropna(axis=0)
    print(f"Dropped {initial_len - len(df_cleaned)} rows with null values")
    
    # drop duplicates
    duplicates = df_cleaned.duplicated().sum()
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"Dropped {duplicates} duplicate rows")
    
    if garbage_time_method in [0, 1, 2]: # if the garbage time method is valid
        if garbage_time_method == 0:
            # option 1: manually define garbage time
            df_cleaned['garbage_time'] = (
                # final 2 min of 1st half
                ((df_cleaned['qtr'] == 2) & (df_cleaned['half_seconds_remaining'] <= 120)) |
                # trailing in the 4th qtr
                ((df_cleaned['qtr'] == 4) & (df_cleaned['score_differential'] < 0)) |
                # down by 21+ at any time
                (df_cleaned['score_differential'] < -21) |
                # up by 21+ in the 2nd half
                ((df_cleaned['qtr'] >= 3) & (df_cleaned['score_differential'] >= 21))
            )
            df_cleaned = df_cleaned.loc[~df_cleaned['garbage_time']].reset_index(drop=True)
            df_cleaned = df_cleaned.drop(columns=['garbage_time'])
            
        elif garbage_time_method == 1:
            # option 2: win probability between 0.1 and 0.9
            before_len = len(df_cleaned)
            df_cleaned = df_cleaned[((df_cleaned['wp'] > 0.1) & (df_cleaned['wp'] < 0.9))]
            print(f"Removed {before_len - len(df_cleaned)} garbage time plays (wp method)")
            
        elif garbage_time_method == 2:  # garbage_time_method == 2
            # option 3: 4th quarter blowouts (> 21 point differential)
            before_len = len(df_cleaned)
            mask_garbage = (df_cleaned['qtr'] == 4) & (df_cleaned['score_differential'].abs() > 21)
            df_cleaned = df_cleaned[~mask_garbage]
            print(f"Removed {before_len - len(df_cleaned)} garbage time plays (4th qtr blowout method)")
        
        else: # if garbage time should not be removed
            pass
    else: # if the garbage time method is not valid
        raise ValueError(f"Invalid garbage time method: {garbage_time_method}")
        

    print(f"Final cleaned dataset: {len(df_cleaned)} rows (from {initial_len} original)")

    return df_cleaned.reset_index(drop=True)

# 4) engineer features by converting data types to categorical
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    engineer features by converting data types.
    
    args:
        df: inputdataframe to engineer features for
        
    Returns:
        DataFrame with engineered features
    """

    df_engineered = df.copy()
    
    # convert down and qtr to categorical for better gradient boosting performance
    df_engineered['down'] = df_engineered['down'].astype('category')
    df_engineered['qtr'] = df_engineered['qtr'].astype('category')
    
    return df_engineered


# complete the preprocessing pipeline, wrap all the steps together
def preprocess_data(df: pd.DataFrame, garbage_time_method: int = 2) -> pd.DataFrame:
    
    print("Starting preprocessing pipeline...")
    
    # Step 1: Feature selection
    df = select_features(df)
    
    # Step 2: Filter play types
    df = filter_play_types(df)
    
    # Step 3: Clean data
    df = clean_data(df, garbage_time_method=garbage_time_method)
    
    # Step 4: Feature engineering
    df = engineer_features(df)
    
    print("Preprocessing complete!")
    return df