import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TARGET_COL, DATE_COL, FORECAST_HORIZON
from src.data_loader import load_and_prepare
from src.feature_engineering import build_features, get_feature_columns
from src.models import RidgeForecaster
from src.evaluation import compute_metrics, time_series_cv_splits
from src.forecast import generate_forecast


def test_endToEndPipeline():
    df = load_and_prepare()
    assert df[TARGET_COL].isnull().sum() == 0

    df_feat = build_features(df.copy())
    all_cols = get_feature_columns(df_feat)
    no_lag = [c for c in all_cols if not c.startswith("lag_") and not c.startswith("roll_")]
    valid = df_feat[df_feat[all_cols].notna().all(axis=1)].reset_index(drop=True)

    model = RidgeForecaster()
    model.fit(valid[no_lag], valid[TARGET_COL])
    forecast = generate_forecast(model, df, FORECAST_HORIZON)

    assert len(forecast) == 90
    assert (forecast["forecast_complaints"] >= 0).all()
    assert forecast[DATE_COL].min() > df[DATE_COL].max()


def test_crossValidation_noLeakage():
    splits = time_series_cv_splits(1000, n_splits=5, test_size=90)
    assert len(splits) == 5
    for tr, te in splits:
        assert tr.max() < te.min()
