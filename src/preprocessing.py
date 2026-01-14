import load as di
import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # general feature trim to remove unrelated features (e.g. probabilities, post play features, etc)
    # before actual feature selection
    cols_to_keep = [
        "yardline_100", # Numeric distance in the number of yards from the opponent's endzone for the posteam.
        "posteam", # String abbreviation for the team with possession.
        "quarter_seconds_remaining",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "game_half", # String indicating which half the play is in, either Half1, Half2, or Overtime.
        "drive", # Numeric drive number in the game.
        "qtr", # Quarter of the game (5 is overtime).
        "down", 
        "goal_to_go", # Binary indicator for whether or not the posteam is in a goal down situation.
        "time", # Time at start of play provided in string format as minutes:seconds remaining in the quarter.
        "ydstogo",
        "play_type",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "posteam_score",
        "defteam_score",
        "score_differential", # Score differential between the posteam and defteam at the start of the play.
        "ep", # estimated expected points with respect to the possession team for the given play.
        "posteam_type", # String indicating whether the posteam team is home or away
        "wp", # Estimated win probabiity for the posteam given the current situation at the start of the given play
        "season"
    ]

    df = df[cols_to_keep]
    return None