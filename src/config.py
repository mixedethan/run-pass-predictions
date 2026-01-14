# feature lists for different models
FEATURES_DECISION = [
    'yardline_100',
    'ydstogo',
    'score_differential',
    'game_seconds_remaining',
    'posteam_timeouts_remaining'
]

FEATURES_PLAY = [
    'down',
    'ydstogo',
    'yardline_100',
    'score_differential',
    'game_seconds_remaining',
    'posteam_timeouts_remaining',
    'qtr'
]

# model hyperparameters
MODEL_PARAMS_DECISION = {
    'n_estimators': [650],
    'learning_rate': [0.1],
    'max_depth': [5],
    'num_leaves': [15]
}

MODEL_PARAMS_PLAY = {
    'n_estimators': [300],
    'learning_rate': [0.03],
    'max_depth': [8],
    'num_leaves': [30]
}

# training configs
RANDOM_STATE = 46
TEST_SEASON = 2024
TIME_SERIES_CV_SPLITS = 3

# data configs
VALID_PLAY_TYPES = ['pass', 'run', 'punt', 'field_goal']
GARBAGE_TIME_METHOD = 2  # 0, 1, or 2 (0 = manually define, 1 = win probability, 2 = 4th quarter blowouts)
