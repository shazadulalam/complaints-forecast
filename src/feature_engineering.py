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