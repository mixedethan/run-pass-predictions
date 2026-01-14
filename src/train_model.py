import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, f1_score, classification_report
import lightgbm as lgb
from typing import Tuple, Dict, Any
import config


# split the data into train and test sets by season
def split_data(df: pd.DataFrame, test_season: int = config.TEST_SEASON) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    split data into train and test sets by season.
    
    args:
        df: preprocessed dataframe
        test_season: season to use as the test set
        
    returns:
        tuple of (train_df, test_df)
    """
    train_master = df[df['season'] < test_season].copy()
    test_master = df[df['season'] == test_season].copy()
    
    print(f"Train set: {len(train_master)} rows (seasons < {test_season})")
    print(f"Test set: {len(test_master)} rows (season == {test_season})")
    
    return train_master, test_master
# return the train and test master dataframes -> (train_master, test_master)

def prepare_decision_data(train_master: pd.DataFrame, test_master: pd.DataFrame, features: list = config.FEATURES_DECISION) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    prepare data for decision model (go for it vs kick on 4th down).
    
    args:
        train_master: training dataframe
        test_master: test dataframe
        features: list of features to use 
        
    returns:
        tuple of (X_train, y_train, X_test, y_test)
    """
    
    
    # filter to 4th down plays
    train_decision = train_master[train_master['down'] == 4].copy()
    test_decision = test_master[test_master['down'] == 4].copy()
    
    # create target: 0 = Kick (punt/fg), 1 = Go (pass/run)
    train_decision['target_go'] = train_decision['play_type'].isin(['pass', 'run']).astype(int)
    test_decision['target_go'] = test_decision['play_type'].isin(['pass', 'run']).astype(int)
    
    X_train = train_decision[features].copy()
    y_train = train_decision['target_go']
    X_test = test_decision[features].copy()
    y_test = test_decision['target_go']
    
    print(f"Decision model - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Decision model - Class distribution (train): {y_train.value_counts().to_dict()}")
    
    return X_train, y_train, X_test, y_test
# return the train and test decision dataframes -> (X_train, y_train, X_test, y_test)

def prepare_play_data(train_master: pd.DataFrame, test_master: pd.DataFrame,features: list = config.FEATURES_PLAY) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    prepare data for play prediction model (run vs pass).
    
    args:
        train_master: training dataframe
        test_master: test dataframe
        features: list of features to use (defaults to FEATURES_PLAY)
        
    returns:
        tuple of (X_train, y_train, X_test, y_test)
    """
    
    # filter to run and pass plays only
    train_play = train_master[train_master['play_type'].isin(['run', 'pass'])].copy()
    test_play = test_master[test_master['play_type'].isin(['run', 'pass'])].copy()
    
    # create target: 0 = run, 1 = pass
    train_play['target_pass'] = (train_play['play_type'] == 'pass').astype(int)
    test_play['target_pass'] = (test_play['play_type'] == 'pass').astype(int)
    
    X_train = train_play[features].copy()
    y_train = train_play['target_pass']
    X_test = test_play[features].copy()
    y_test = test_play['target_pass']
    
    print(f"Play model - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Play model - Class distribution (train): {y_train.value_counts().to_dict()}")
    
    return X_train, y_train, X_test, y_test
# return the train and test play dataframes -> (X_train, y_train, X_test, y_test)

def train_decision_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, param_grid: Dict[str, list] = config.MODEL_PARAMS_DECISION) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """
    train the decision model (go for it vs kick on 4th down).
    
    args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        param_grid: Grid search parameters (defaults to notebook values)
        
    Returns:
        tuple of (trained_model, evaluation_metrics)
    """
    
    print("Training Decision Model...")
    
    # time series cross-validation
    tscv = TimeSeriesSplit(n_splits=config.TIME_SERIES_CV_SPLITS)
    
    # base model -> unbalanced since going on 4th is rare
    lgb_decision = lgb.LGBMClassifier(random_state=config.RANDOM_STATE, is_unbalance=True, verbose=-1)
    
    grid_decision = GridSearchCV(
        estimator=lgb_decision,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1',  # imbalanced set, focus on f1
        verbose=1,
        n_jobs=-1
    )
    
    print('Training model with Grid Search...')
    grid_decision.fit(X_train, y_train)
    print('Training complete!')
    print(f'Best Parameters: {grid_decision.best_params_}')
    
    best_model = grid_decision.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"Model 1's Accuracy: {metrics['accuracy']:.3f}")
    print(f"Model 1's Precision: {metrics['precision']:.3f}")
    print(f"Model 1's Recall: {metrics['recall']:.3f}")
    print(f"Model 1's F1 Score: {metrics['f1']:.3f}")
    
    return best_model, metrics


def train_play_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, param_grid: Dict[str, list] = config.MODEL_PARAMS_PLAY) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """
    train the play prediction model (run vs pass).
    
    args:
        X_train: training features
        y_train: training target
        X_test: test features
        y_test: test target
        param_grid: grid search parameters
        
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """

    print("Training Play Prediction Model...")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=config.TIME_SERIES_CV_SPLITS)
    
    lgb_play = lgb.LGBMClassifier(random_state=config.RANDOM_STATE, verbose=-1)
    
    grid_play = GridSearchCV(
        estimator=lgb_play,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_log_loss',  # use neg_log_loss because we want to focus on probability calibration
        verbose=1,
        n_jobs=-1
    )
    
    print('Training model with Grid Search...')
    grid_play.fit(X_train, y_train)
    print('Training complete!')
    print(f'Best Parameters: {grid_play.best_params_}')
    
    best_model = grid_play.best_estimator_
    
    # evaluate
    y_pred_prob = best_model.predict_proba(X_test)
    y_pred = (y_pred_prob[:, 1] > 0.5).astype(int) # convert to 0 & 1
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'baseline_accuracy': y_test.mean()  # always guessing pass
    }
    
    print(f"Model's Accuracy: {metrics['accuracy']:.3f}")
    print(f"Model's Precision: {metrics['precision']:.3f}")
    print(f"Model's Recall: {metrics['recall']:.3f}")
    print(f"Model's F1 Score: {metrics['f1']:.3f}")
    print(f"Dummy Accuracy (Always guessing Pass): {metrics['baseline_accuracy']:.3f}")
    
    return best_model, metrics


def save_model(model: Any, filepath: str) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f'Model successfully saved to {filepath}')

# main training pipeline function call
def train_pipeline(df: pd.DataFrame, test_season: int = config.TEST_SEASON, save_models: bool = True) -> Dict[str, Any]:
    """
    complete training pipeline for both models.
    
    Args:
        df: preprocessed dataframe
        models_dir: directory to save models (defaults to run-pass-predictions/models)
        test_season: season to use as the test set
        save_models: whether to save models to disk
        
    returns:
        dictionary containing the two trained models and their metrics
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1] # /run-pass-predictions
    models_dir = PROJECT_ROOT / "models" # /run-pass-predictions/models
    
    models_dir = Path(models_dir) # turn the models_dir into a Path object
    models_dir.mkdir(parents=True, exist_ok=True) # create the models directory if it doesn't exist
    
    # split data into train and test sets
    train_master, test_master = split_data(df, test_season=test_season)
    
    # train the decision model
    X_train_dec, y_train_dec, X_test_dec, y_test_dec = prepare_decision_data(train_master, test_master)
    decision_model, decision_metrics = train_decision_model(X_train_dec, y_train_dec, X_test_dec, y_test_dec)
    
    if save_models == True:
        save_model(decision_model, models_dir / "decision_model.pkl")
    
    # Train Play Model
    X_train_play, y_train_play, X_test_play, y_test_play = prepare_play_data(train_master, test_master)
    play_model, play_metrics = train_play_model(X_train_play, y_train_play, X_test_play, y_test_play)
    
    if save_models:
        save_model(play_model, models_dir / "play_model.pkl")
    
    return {
        'decision_model': decision_model,
        'decision_metrics': decision_metrics,
        'play_model': play_model,
        'play_metrics': play_metrics
    }
