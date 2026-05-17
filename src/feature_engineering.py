import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import TARGET_COL, DATE_COL, EXOG_COLS


# 

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df["trend_idx"] = (df[DATE_COL] - df[DATE_COL].min()).dt.days

    # fit train on target col which will be  training data
    mask = df[TARGET_COL].notna()
    if mask.sum() > 0:
        lr = LinearRegression()
        lr.fit(df.loc[mask, ["trend_idx"]], df.loc[mask, TARGET_COL])
        df["trend_linear"] = lr.predict(df[["trend_idx"]])
    else:
        df["trend_linear"] = df["trend_idx"] * 0.036 + 60

    return df



def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    
    dt = df[DATE_COL]  
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["day_of_year"] = dt.dt.dayofyear

    # fourier terms applying on weekly period
    for k in [1, 2]:
        df[f"week_sin_{k}"] = np.sin(2 * np.pi * k * df["day_of_week"] / 7)
        df[f"week_cos_{k}"] = np.cos(2 * np.pi * k * df["day_of_week"] / 7)

    # fourier terms applying on annual period
    for k in [1, 2, 3]:
        df[f"year_sin_{k}"] = np.sin(2 * np.pi * k * df["day_of_year"] / 365.25)
        df[f"year_cos_{k}"] = np.cos(2 * np.pi * k * df["day_of_year"] / 365.25)

    return df


LAG_DAYS = [1, 2, 3, 7, 14, 28]

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)
    return df


ROLL_WINDOWS = [7, 14, 28]

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in ROLL_WINDOWS:
        rolled = df[TARGET_COL].shift(1).rolling(w, min_periods=1)
        df[f"roll_mean_{w}"] = rolled.mean()
        df[f"roll_std_{w}"] = rolled.std()
    return df


# exogenous column identification

def validate_exog(df: pd.DataFrame) -> pd.DataFrame:
    for col in EXOG_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_trend_features(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = validate_exog(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {DATE_COL, TARGET_COL, "row_id", "centered_7d_mean"}
    return [c for c in df.columns if c not in exclude]
