from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
RAW_FILE = DATA_DIR / "Principle_Data_Scientist_Tech_Assessment.xlsx"
SHEET_NAME = "daily records"

FORECAST_HORIZON = 90
TARGET_COL = "complaints"
DATE_COL = "date"

# exogenous features

EXOG_COLS = [
    "is_weekend",
    "bank_holiday_flag",
    "staffing_level_fte",
    "backlog_days",
    "media_mentions",
    "channel_mix_index",
]

# time series cross validation

CV_N_SPLITS = 5
CV_TEST_SIZE = 90
CV_GAP = 0

# LightGBM hyperparameters 
# heavier regularisation prevents overfitting to noise while the trend_linear feature handles extrapolation

LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 5,
    "num_leaves": 20,
    "subsample": 0.75,
    "colsample_bytree": 0.7,
    "min_child_samples": 15,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "random_state": 42,
    "verbose": -1,
}

SEED = 42