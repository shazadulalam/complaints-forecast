import pandas as pd
import numpy as np
from src.config import RAW_FILE, SHEET_NAME, TARGET_COL, DATE_COL


def load_raw() -> pd.DataFrame:
    
    df = pd.read_excel(RAW_FILE, sheet_name=SHEET_NAME)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    return df


def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    
    full_idx = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    df = df.set_index(DATE_COL).reindex(full_idx)
    df.index.name = DATE_COL
    
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    df = df.ffill().bfill()
    df = df.reset_index()

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    
    df[TARGET_COL] = df[TARGET_COL].interpolate(method="linear").round()
    
    #non-negetive column
    df[TARGET_COL] = df[TARGET_COL].clip(lower=0)

    return df


def load_and_prepare() -> pd.DataFrame:
    
    df = load_raw()
    df = fill_missing_dates(df)
    df = clean(df)
    
    return df