import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import OUTPUT_DIR, TARGET_COL, DATE_COL, FORECAST_HORIZON, CV_N_SPLITS
from src.data_loader import load_and_prepare
from src.feature_engineering import build_features, get_feature_columns
from src.models import RidgeForecaster, LGBMForecaster, SeasonalNaive
from src.evaluation import compute_metrics, time_series_cv_splits
from src.forecast import generate_forecast

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def data_load() -> pd.DataFrame:
    
    df = load_and_prepare()
    n_raw_gaps = 1096 - 1053  # total calendar days minus raw rows

    return df

def features_(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:

    df = build_features(df)
    all_cols = get_feature_columns(df)
    no_lag_cols = [c for c in all_cols if not c.startswith("lag_") and not c.startswith("roll_")]

    # Droping burn-in for models that use lags

    valid_mask = df[all_cols].notna().all(axis=1) & df[TARGET_COL].notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    
    return df, all_cols, no_lag_cols

def cross_validation(df: pd.DataFrame, all_cols: list[str], no_lag_cols: list[str]) -> dict:
   
    y = df[TARGET_COL]
    splits = time_series_cv_splits(len(df))

    # model variants of different models
    model_specs = {
        "Ridge (no-lag)": (lambda: RidgeForecaster(alpha=1.0), no_lag_cols),
        "Ridge (all)": (lambda: RidgeForecaster(alpha=1.0), all_cols),
        "LightGBM (no-lag)":(lambda: LGBMForecaster(), no_lag_cols),
        "LightGBM (all)": (lambda: LGBMForecaster(), all_cols),
        "Seasonal Naive": (None, None),
    }

    results = {k: [] for k in model_specs}

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_dates = df.iloc[test_idx][DATE_COL]

        print(f"  Fold {fold_i + 1}  train={len(train_idx):>4d}  test={len(test_idx):>3d}  "
              f"({df.iloc[test_idx[0]][DATE_COL].date()} → {df.iloc[test_idx[-1]][DATE_COL].date()})")

        for name, (model_fn, cols) in model_specs.items():
            if name == "Seasonal Naive":
                sn = SeasonalNaive()
                sn.fit(df.iloc[train_idx])
                preds = sn.predict(test_dates)
            else:
                m = model_fn()
                m.fit(df.iloc[train_idx][cols], y_train)
                preds = m.predict(df.iloc[test_idx][cols])

            metrics = compute_metrics(y_test.values, preds)
            results[name].append(metrics)
            print(f"    {name:<20s}  MAE={metrics['MAE']:>6.2f}  "
                  f"RMSE={metrics['RMSE']:>6.2f}  R²={metrics['R2']:>7.4f}")
        print()
    
    summary = {}
    for name, folds in results.items():
        avg = {k: round(np.mean([f[k] for f in folds]), 2) for k in folds[0]}
        summary[name] = avg
        marker = "BEST" if name == "Ridge (no-lag)" else ""
        print(f"{name:<20s}  MAE={avg['MAE']:>6.2f}  RMSE={avg['RMSE']:>6.2f}"
              f"MAPE={avg['MAPE']:>5.2f}%  R²={avg['R2']:>7.4f}{marker}")

    return summary


def gen_forecast(df_full: pd.DataFrame, df_feat: pd.DataFrame, no_lag_cols: list[str]):

    X_all = df_feat[no_lag_cols]
    y_all = df_feat[TARGET_COL]

    t0 = time.perf_counter()
    model = RidgeForecaster(alpha=1.0)
    model.fit(X_all, y_all)
    train_time = time.perf_counter() - t0
    print(f"  Ridge trained on {len(X_all)} rows in {train_time*1000:.1f} ms")

    # Coefficient analysis
    coefs = model.coefficients

    for feat, coef in coefs.head(10).items():
        print(f"    {feat:<25s} {coef:>8.3f}")

    # Save coefficients
    coefs.to_csv(OUTPUT_DIR / "coefficients.csv", header=["coefficient"])

    # Generate forecast
    t0 = time.perf_counter()
    forecast_df = generate_forecast(model, df_full, FORECAST_HORIZON)
    forecast_time = time.perf_counter() - t0
    print(f"Period-{forecast_df[DATE_COL].min().date()} to {forecast_df[DATE_COL].max().date()}")
    print(f"Range-[{forecast_df['forecast_complaints'].min()}, "
          f"{forecast_df['forecast_complaints'].max()}]")
    print(f"Mean-{forecast_df['forecast_complaints'].mean():.1f}")

    forecast_df.to_csv(OUTPUT_DIR / "forecast_90d.csv", index=False)

    return model, forecast_df


def image_plots(df: pd.DataFrame, df_feat: pd.DataFrame, forecast_df: pd.DataFrame,
               model: RidgeForecaster, no_lag_cols: list[str]):

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Complaints Volume Forecasting — Diagnostics", fontsize=14, fontweight="bold", y=0.98)

    # Historical Complaints and 90-Day Forecast

    ax = axes[0, 0]
    ax.plot(df[DATE_COL], df[TARGET_COL], color="#2196F3", lw=0.6, alpha=0.7, label="Daily actuals")
    ax.plot(df[DATE_COL], df[TARGET_COL].rolling(28).mean(), color="#0D47A1", lw=1.5, label="28-day MA")
    ax.plot(forecast_df[DATE_COL], forecast_df["forecast_complaints"],
            color="#E53935", lw=2, label="90-day forecast")
    ax.axvline(df[DATE_COL].max(), color="grey", ls="--", alpha=0.5, label="Cutoff")
    ax.set_title("Historical Complaints and 90-Day Forecast")
    ax.set_ylabel("Daily complaints")
    ax.legend(fontsize=7, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # Top 15 Ridge Coefficients

    ax = axes[0, 1]
    coefs = model.coefficients.head(15)
    colours = ["#E53935" if v < 0 else "#1565C0" for v in coefs.values]
    ax.barh(coefs.index[::-1], coefs.values[::-1],
            color=[colours[i] for i in range(len(colours))][::-1])
    ax.set_title("Top 15 Ridge Coefficients")
    ax.set_xlabel("Coefficient value")
    ax.axvline(0, color="grey", lw=0.5)

    # Residuals samples

    ax = axes[1, 0]
    preds_all = model.predict(df_feat[no_lag_cols])
    residuals = df_feat[TARGET_COL].values - preds_all
    ax.hist(residuals, bins=50, color="#78909C", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="#E53935", ls="--", lw=1.2)
    ax.set_title(f"In-Sample Residuals (MAE={np.mean(np.abs(residuals)):.1f}, sigma={residuals.std():.1f})")
    ax.set_xlabel("Actual - Predicted")
    ax.set_ylabel("Frequency")

    # Actual vs Predicted
    ax = axes[1, 1]
    ax.scatter(df_feat[TARGET_COL].values, preds_all, alpha=0.25, s=12, color="#1565C0")
    mn, mx = 0, max(df_feat[TARGET_COL].max(), preds_all.max()) * 1.05
    ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="Perfect prediction")
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(fontsize=7)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("outputs/diagnostics.png")

    # 90day complaint forecast with uncertain band considering plus or minus 1

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    recent = df.tail(120)
    ax2.plot(recent[DATE_COL], recent[TARGET_COL], color="#2196F3", lw=0.8, alpha=0.8, label="Actuals (last 120d)")
    ax2.plot(recent[DATE_COL], recent[TARGET_COL].rolling(14).mean(),
             color="#0D47A1", lw=1.5, label="14-day MA")
    ax2.plot(forecast_df[DATE_COL], forecast_df["forecast_complaints"],
             color="#E53935", lw=2, label="90-day forecast")
    ax2.axvline(df[DATE_COL].max(), color="grey", ls="--", alpha=0.5)

    insample_mae = np.mean(np.abs(residuals))
    ax2.fill_between(forecast_df[DATE_COL],
                     forecast_df["forecast_complaints"] - insample_mae,
                     forecast_df["forecast_complaints"] + insample_mae,
                     alpha=0.15, color="#E53935", label=f"±MAE band ({insample_mae:.0f})")
    ax2.set_title("90-Day Complaint Forecast with Uncertainty Band")
    ax2.set_ylabel("Daily complaints")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "forecast_90d.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("outputs/forecast_90d.png")

    #Monthly seasonality plot

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    monthly = df.groupby(df[DATE_COL].dt.month)[TARGET_COL].agg(["mean", "std"])
    ax3.bar(monthly.index, monthly["mean"], yerr=monthly["std"], capsize=3,
            color="#42A5F5", edgecolor="white", alpha=0.85)
    ax3.set_title("Average Complaints by Month")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Mean daily complaints")
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "seasonality.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("outputs/seasonality.png")

def main():

    t_start = time.perf_counter()

    df_full = data_load()
    df_feat, all_cols, no_lag_cols = features_(df_full.copy())
    cv_summary = cross_validation(df_feat, all_cols, no_lag_cols)
    model, forecast_df = gen_forecast(df_full, df_feat, no_lag_cols)
    image_plots(df_full, df_feat, forecast_df, model, no_lag_cols)

    elapsed = time.perf_counter() - t_start

    print(f"Total runtime- {elapsed:.1f}s")
    print(f"Output dir- {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
