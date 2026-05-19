# Complaint Volume Forecasting

Predicting daily complaint volumes for the next ninety days after a specified datetime cutoff is the main goal of the `complaint-forecast` project. The pipeline generates an accurate and computationally effective forecast through the identification of trend, seasonality, and external causes using the analysis of past complaint patterns in combination of exogenous operational parameters.

## Task Summary

At the moment, complaint volumes are measured reactively without projections. The goal of this project is to produce a fully operational forecasting pipeline that takes around three years' worth of daily complaint data, builds useful time-series characteristics, rigorously cross-validates many potential models, and produces a daily prediction for ninety days. An Excel spreadsheet with daily records of personnel levels, backlog days, media mentions, and channel mix indicators is the source of the data. The planning of capacity, assessment resource allocation, and operational judgment may all benefit from the comprehensive, daily estimate of anticipated complaint volumes provided by the converted and modeled data.

## Project Details

### 1. **Data Loading & Cleaning:**
- The Excel source contains 1,053 raw daily records from 2023-01-01 to 2025-12-31.
- A continuous daily index is created by filling in the 43 missing calendar days.
- Exogenous gaps are forward-filled, and ten missing target values are interpolated linearly.

### 2. **Exploratory Data Analysis:**
- Distribution analysis, correlation heatmaps, seasonal decomposition, stationarity tests (ADF), ACF/PACF plots, trend quantification, outlier identification, and a six-model comparison are all covered in a 25-cell Jupyter notebook.
- Significant results include a linear trend of +13.1 complaints per year, an annual seasonality of ±15 from mean, and weak lag correlations (<0.35).

### 3. **Feature Engineering:**
- Calendar features: day_of_week, month, quarter, week_of_year, day_of_year
- Fourier cyclical encodings: sin/cos at weekly (k=1,2) and annual (k=1,2,3) periods
- Trend features: integer day-index and OLS linear projection
- Lag features: lag_1, lag_7, lag_14, lag_28 (tested but excluded from final model)
- Rolling windows: 7/14/28-day mean and std (tested but excluded from final model)
- Exogenous: staffing_level_fte, backlog_days, media_mentions, channel_mix_index

### 4. **Model Selection & Evaluation:**
- Six model variants evaluated via expanding-window time-series CV (5 folds, 90-day test windows).
- Ridge Regression (no-lag) selected as the best model based on lowest MAE across all folds.

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| Ridge (no-lag) | 21.02 | 26.37 | 30.03% | -0.03 |
| Ridge (all features) | 21.50 | 26.93 | 30.53% | -0.07 |
| LightGBM (no-lag) | 23.44 | 29.12 | 32.06% | -0.25 |
| HistGradientBoosting (no-lag) | 24.24 | 30.16 | 33.72% | -0.33 |
| HistGradientBoosting (all) | 25.24 | 31.13 | 34.72% | -0.43 |

### 5. **Models Considered but Not Selected:**
- **Prophet:** Same Fourier and trend decomposition achieves like the selected RidgeForecaster. 
- **ARIMA/SARIMAX:** Requires stationarity transforms, handles exogenous regressors awkwardly, slower for 90-step forecasting.
- **XGBoost:** Same tree-based limitations as HistGB. No advantage for this dataset size.

### 6. **Forecasting:**
- The complete 90-day prediction is produced in a single pass (direct, non-recursive) because the final model does not include lag characteristics. 
- For Q1 2026, the forecast range is 106–126 complaints per day, with a mean of ≈ 118.

## Getting Started

### Pre-requisites
- Python 3.10 or later
- pip or conda package manager
- The Excel data file placed inside `data/` directory

### Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone git@github.com:shazadulalam/complaints-forecast.git
   cd complaint-forecasting
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Project Structure

```bash
complaints-forecast/
|-- requirements.txt
|-- main.py
|-- Dockerfile
|-- docker-compose.yml
|-- README.md
|-- .gitignore
|-- .github/
|   -- workflows/
|       -- ci.yml
|-- notebook/
|   -- eda_notebook.ipynb
|-- data/
|   -- Principle_Data_Scientist_Tech_Assessment.xlsx
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- feature_engineering.py
|   |-- models.py
|   |-- evaluation.py
|   |-- forecast.py
|-- tests/
|   |-- __init__.py
|   |-- test_pipeline.py
`-- outputs/
    |-- forecast_90d.csv
    |-- coefficients.csv
    |-- diagnostics.png
    |-- forecast_90d.png
    |-- seasonality.png
```
#### Please make sure to change your data file location.

### Usage

1. **Run the forecasting pipeline:**
   ```bash
   python main.py
   ```
   This will execute the full pipeline- data loading, feature engineering, cross-validation, model training, 90-day forecast generation, and diagnostic plot creation. All outputs are saved to the `outputs/` directory.

2. **Run the test suite:**
   ```bash
   pytest tests/test_pipeline.py -v
   ```
   Two smoke tests verify the end-to-end pipeline produces valid forecasts and that the CV splits contain no future data leakage.

3. **Run the EDA notebook:**
   ```bash
   jupyter notebook eda_notebook.ipynb
   ```

4. **Run with Docker:**
   ```bash
   docker compose up --build
   ```
   Outputs will be added to `./outputs/` on the host machine.

### What to Expect

Running `python main.py` will print step-by-step progress across 5 stages and complete in under 5 seconds. The pipeline produces:
- `forecast_90d.csv` — 90 rows of daily complaint predictions for Q1 2026
- `coefficients.csv` — Ridge model coefficients showing feature contributions
- `diagnostics.png` — 4-panel chart (time series, coefficients, residuals, actual vs predicted)
- `forecast_90d.png` — forecast close-up with ±MAE uncertainty band
- `seasonality.png` — monthly complaint pattern

### CI/CD

`.github/workflows/ci.yml` defines a GitHub Actions workflow. For Python 3.10, 3.11, and 3.12, it automatically runs tests and the entire pipeline on each push to `main`.
