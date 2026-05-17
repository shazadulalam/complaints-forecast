import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb

from src.config import LGBM_PARAMS, SEED


#Ridge regression on calendar + trend + exogenous features, L2 regularisation prevents overfitting on the 36 features
#while preserving the linear trend extrapolation that is critical for multi-step forecasting.

class RidgeForecaster:
    
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha, random_state=SEED)
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeForecaster":
        self._feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(np.round(preds), 0, None)

    @property
    def feature_importance(self) -> pd.Series:
        """Absolute coefficient magnitudes as a proxy for importance."""
        return pd.Series(
            np.abs(self.model.coef_),
            index=self._feature_names,
        ).sort_values(ascending=False)

    @property
    def coefficients(self) -> pd.Series:
        return pd.Series(self.model.coef_, index=self._feature_names).sort_values(
            key=abs, ascending=False
        )


class LGBMForecaster:

    def __init__(self, params: dict | None = None):
        self.params = params or LGBM_PARAMS
        self.model = lgb.LGBMRegressor(**self.params)
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMForecaster":
        self._feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(np.round(preds), 0, None)

    @property
    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)


class SeasonalNaive:

    def __init__(self):
        self._weekday_avg: dict[int, float] = {}

    def fit(self, df: pd.DataFrame, target_col: str = "complaints") -> "SeasonalNaive":
        recent = df.tail(28)
        self._weekday_avg = (
            recent.groupby(recent["date"].dt.dayofweek)[target_col].mean().to_dict()
        )
        return self

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return np.array([round(self._weekday_avg.get(d.dayofweek, 0)) for d in dates])
