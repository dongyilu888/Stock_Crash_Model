import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

# Constants
DATA_DIR = 'data'
FEATURES_FILE = os.path.join(DATA_DIR, 'features.csv')
RESULTS_DIR = 'results'
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'predictions_gbm.csv') # Distinct file
METRICS_FILE = os.path.join(RESULTS_DIR, 'metrics_gbm.csv')
MODEL_FILE = os.path.join(RESULTS_DIR, 'latest_model_gbm.pkl')
IMPORTANCE_FILE = os.path.join(RESULTS_DIR, 'importance_gbm.pkl')

FEATURES = [
    'CAPE', 'Earnings_Yield', 
    'Inflation_12m', 'Unemployment_Rate', 'Unemployment_Change_12m', 
    'Term_Spread', 'Credit_Spread', 'Real_Rate', 
    'Commodity_Ret_12m', 'S&P_Ret_12m', 'S&P_Vol_12m'
]

TARGET = 'Target_Crash_12m'

def somers_d_score(y_true, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
        return 2 * auc - 1
    except:
        return np.nan

def train_gbm():
    print("Training Gradient Boosting model (HistGradientBoosting)...")
    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=[0])
    df = df.sort_index()
    
    # We use expanding window manually to generate full history predictions similar to RF
    # TimeSeriesSplit is great for cross-validation metrics, but we want a "Live Backtest" curve.
    
    start_date = pd.Timestamp('1960-01-01')
    predictions = []
    years = pd.date_range(start=start_date, end=df.index.max(), freq='YS')
    
    if df.index.min() > start_date:
        years = pd.date_range(start=df.index.min() + pd.DateOffset(years=5), end=df.index.max(), freq='YS')

    # Model
    # HistGradientBoosting handles NaNs natively! No imputer needed.
    model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=100, max_depth=5, random_state=42)
    
    print(f"Backtesting from {years[0].year} to {years[-1].year}...")
    
    for dt in years:
        train_mask = df.index < dt
        test_mask = (df.index >= dt) & (df.index < dt + pd.DateOffset(years=1))
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if test_df.empty or len(train_df) < 50:
            continue
            
        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]
        X_test = test_df[FEATURES]
        
        # Fit
        model.fit(X_train, y_train)
        
        # Predict
        probs = model.predict_proba(X_test)[:, 1]
        
        res = pd.DataFrame({
            'Crash_Prob': probs,
            'Target': test_df[TARGET]
        }, index=test_df.index)
        predictions.append(res)
        
    if not predictions:
        print("No predictions generated.")
        return
        
    full_preds = pd.concat(predictions)
    full_preds.index.name = 'Date'
    full_preds.to_csv(PREDICTIONS_FILE)
    print(f"Predictions saved to {PREDICTIONS_FILE}")
    
    overall_sd = somers_d_score(full_preds['Target'], full_preds['Crash_Prob'])
    print(f"Overall GBM Out-of-Sample Somers' D: {overall_sd:.4f}")
    
    # Rolling Metrics
    metrics = []
    window_size = 60
    for end_idx in range(window_size, len(full_preds)):
        window = full_preds.iloc[end_idx-window_size:end_idx]
        if window['Target'].nunique() > 1:
            score = somers_d_score(window['Target'], window['Crash_Prob'])
        else:
            score = np.nan
        metrics.append({'Date': full_preds.index[end_idx], 'Rolling_SomersD': score})
    
    metrics_df = pd.DataFrame(metrics).set_index('Date')
    metrics_df.to_csv(METRICS_FILE)
    
    # Final Model
    print("Training final GBM model on full history...")
    X_full = df[FEATURES]
    y_full = df[TARGET]
    model.fit(X_full, y_full)
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
        
    # Interpretability: Permutation Importance
    # SHAP for HistGradientBoosting is not as straightforward directly with TreeExplainer in some versions,
    # or requires specific wrapper. Permutation importance is model-agnostic and robust.
    print("Calculating Permutation Importance...")
    # Use valid set (or recent data)
    r = permutation_importance(model, X_full, y_full, n_repeats=10, random_state=42, n_jobs=-1)
    
    imp_data = {
        'importances_mean': r.importances_mean,
        'features': FEATURES,
        'method': 'permutation'
    }
    
    with open(IMPORTANCE_FILE, 'wb') as f:
        pickle.dump(imp_data, f)
        
    print("GBM Training complete.")

if __name__ == "__main__":
    train_gbm()
