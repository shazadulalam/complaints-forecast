import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZON, TARGET_COL, DATE_COL, EXOG_COLS
from src.feature_engineering import build_features, get_feature_columns


# Separating the UK bank holidays

_BANK_HOLIDAYS_2026 = {
    pd.Timestamp("2026-01-01"),
    pd.Timestamp("2026-04-03"),
    pd.Timestamp("2026-04-06"),
    pd.Timestamp("2026-05-04"),
    pd.Timestamp("2026-05-25"),
}


def _build_future_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    
    last_date = df[DATE_COL].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    future = pd.DataFrame({DATE_COL: future_dates})
    future["is_weekend"] = future[DATE_COL].dt.dayofweek.isin([5, 6]).astype(int)
    future["bank_holiday_flag"] = future[DATE_COL].isin(_BANK_HOLIDAYS_2026).astype(int)

    last_row = df.iloc[-1]
    for col in EXOG_COLS:
        if col not in ("is_weekend", "bank_holiday_flag"):
            future[col] = last_row.get(col, np.nan)

    future["media_mentions"] = int(df["media_mentions"].tail(28).median())
    future[TARGET_COL] = np.nan
    future["row_id"] = np.nan
    future["centered_7d_mean"] = np.nan
    return future

# model does not use lag feature here, building all 90 future rows at once, feature engineering and predicting in a go

def generate_forecast(
    model,
    df_history: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    
    future = _build_future_frame(df_history, horizon)
    combined = pd.concat([df_history, future], ignore_index=True)

    # Build features on the combined frame (trend_linear extrapolates)
    combined = build_features(combined)

    # Get only the no-lag feature columns
    feature_cols = get_feature_columns(combined)
    # Exclude lag and rolling features
    no_lag_cols = [c for c in feature_cols if not c.startswith("lag_") and not c.startswith("roll_")]

    history_len = len(df_history)
    X_future = combined.iloc[history_len:][no_lag_cols]

    preds = model.predict(X_future)

    forecast_df = pd.DataFrame({
        DATE_COL: combined.iloc[history_len:][DATE_COL].values,
        "forecast_complaints": preds.astype(int),
    })
    return forecast_df.reset_index(drop=True)
