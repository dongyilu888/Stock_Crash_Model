# S&P 500 Crash Prediction Model

A data science project to predict >20% market crashes using historical signals (1929-Present).
Includes an expanding window backtesting framework and an interactive Streamlit dashboard.

## Overview
- **Goal**: Estimate probability of a crash (Drawdown > 20%) in the next 12 months.
- **Data**: Shiller Data (Valuation/Rates/Earnings since 1871), FRED Data (Macro/Credit since 1920s/1950s).
- **Model**: Random Forest Classifier validated specifically via expanding window (preventing lookahead bias).
- **Metric**: Somers' D (Out-of-sample).

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   make setup
   ```
   Or manually: `pip install -r requirements.txt`.

## Usage

### 1. Run Data Pipeline & Training
To download data, process it, engineer features, and train the model in one go:
```bash
make pipeline
```
This executes:
- `data/fetch_data.py`: Downloads Shiller and FRED data.
- `data/process_data.py`: Merges and cleans datasets into `data/processed.csv`.
- `features/build_features.py`: Creates `data/features.csv` with 30+ indicators.
- `model/train.py`: Runs backtest (1960-2025), saves predictions to `results/`.

### 2. Launch Dashboard
Visualize the results:
```bash
make run
```
Or:
```bash
streamlit run app/dashboard.py
```

## Dashboard Features
- **Current Risk**: Gauge chart showing today's estimated crash probability.
- **Drivers**: Feature Importance (Global) or SHAP values (if compatible) showing why risk is high/low.
- **Timeline**: Historical view of Crash Probability vs S&P 500 crashes.
- **Model Performance**: Rolling Somers' D score to spot regimes where the model performs well or poorly.

## Key Files
- `data/`: Data ingestion scripts.
- `features/`: Feature engineering logic.
- `model/`: Training and evaluation script.
- `app/`: Streamlit source code.
