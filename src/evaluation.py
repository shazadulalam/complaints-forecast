import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
)

from src.config import CV_N_SPLITS, CV_TEST_SIZE, CV_GAP

#return dict of regression metrics relevant to forecasting and counting"""

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "MAE": round(mean_absolute_error(yt, yp), 2),
        "RMSE": round(np.sqrt(mean_squared_error(yt, yp)), 2),
        "MdAE": round(median_absolute_error(yt, yp), 2),
        "MAPE": round(mean_absolute_percentage_error(yt, yp) * 100, 2),
        "R2": round(r2_score(yt, yp), 4),
    }


def time_series_cv_splits(
    n_samples: int,
    n_splits: int = CV_N_SPLITS,
    test_size: int = CV_TEST_SIZE,
    gap: int = CV_GAP,
) -> list[tuple[np.ndarray, np.ndarray]]:

    min_train = max(180, n_samples - n_splits * test_size - gap)
    splits = []
    step = (n_samples - min_train - test_size) // max(n_splits - 1, 1)

    for i in range(n_splits):
        train_end = min_train + i * step
        test_start = train_end + gap
        test_end = min(test_start + test_size, n_samples)
        if test_end <= test_start:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits
