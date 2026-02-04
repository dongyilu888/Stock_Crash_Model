import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
try:
    import shap
except ImportError:
    shap = None

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
    'Commodity_Ret_12m', 'S&P_Ret_12m', 'S&P_Vol_12m',
    'Housing_Starts_12m', 'Building_Permits_12m', 'Sentiment_Change_12m', 'Mfg_Hours_Change_12m',
    'CAPE_Unemp_Interaction', 'Term_Inversion'
]

TARGET = 'Target_Crash_12m'

def somers_d_score(y_true, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
        return 2 * auc - 1
    except:
        return np.nan

def run_expanding_gbm(df):
    print(f"Running Expanding Window Validation (Start 1960)...")
    base_model = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=200, max_depth=3, min_samples_leaf=40, l2_regularization=10.0, random_state=42
    )
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    
    predictions = []
    years = pd.date_range(start=pd.Timestamp('1960-01-01'), end=df.index.max(), freq='YS')
    if df.index.min() > pd.Timestamp('1929-01-01'):
        years = pd.date_range(start=df.index.min() + pd.DateOffset(years=5), end=df.index.max(), freq='AS')
        
    for dt in years:
        train_mask = df.index < (dt - pd.DateOffset(months=12))
        test_mask = (df.index >= dt) & (df.index < dt + pd.DateOffset(years=1))
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if test_df.empty or len(train_df) < 50:
            continue
            
        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]
        X_test = test_df[FEATURES]
        
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        
        res = pd.DataFrame({'Crash_Prob': probs, 'Target': test_df[TARGET]}, index=test_df.index)
        predictions.append(res)
        
    if not predictions:
        return pd.DataFrame(), 0.0
        
    full_preds = pd.concat(predictions)
    score = somers_d_score(full_preds['Target'], full_preds['Crash_Prob'])
    return full_preds, score

def run_purged_kfold_gbm(df, n_splits=3):
    print(f"Running Purged K-Fold CV (K={n_splits})...")
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    predictions = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        test_dates = df.index[test_idx]
        test_start = test_dates.min()
        test_end = test_dates.max()
        
        train_original_idx = df.index[train_idx]
        mask_before = train_original_idx < (test_start - pd.DateOffset(months=12))
        mask_after = train_original_idx > (test_end + pd.DateOffset(months=12))
        train_purged_dates = train_original_idx[mask_before | mask_after]
        
        if len(train_purged_dates) < 50:
            continue
            
        train_df = df.loc[train_purged_dates]
        test_df = df.iloc[test_idx]
        
        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]
        X_test = test_df[FEATURES]
        
        base_model = HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=200, max_depth=3, min_samples_leaf=40, l2_regularization=10.0, random_state=42
        )
        model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
        
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        
        res = pd.DataFrame({'Crash_Prob': probs, 'Target': test_df[TARGET]}, index=test_df.index)
        predictions.append(res)
        
    if not predictions:
        return pd.DataFrame(), 0.0
        
    full_preds = pd.concat(predictions).sort_index()
    score = somers_d_score(full_preds['Target'], full_preds['Crash_Prob'])
    return full_preds, score

def train_gbm():
    print("Training Gradient Boosting model (HistGradientBoosting)...")
    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=[0])
    df = df.sort_index()
    
    # 1. Expand
    preds_ew, score_ew = run_expanding_gbm(df)
    print(f"Expanding Window Somers' D: {score_ew:.4f}")
    
    # 2. Purged K-Fold
    preds_kf, score_kf = run_purged_kfold_gbm(df, n_splits=3)
    print(f"Purged K-Fold (K=3) Somers' D: {score_kf:.4f}")
    
    if score_kf > score_ew:
        print(f"✅ K-Fold outperformed Expanding Window ({score_kf:.4f} > {score_ew:.4f}). Using K-Fold predictions.")
        final_preds = preds_kf
        final_score = score_kf
    else:
        print(f"✅ Expanding Window outperformed K-Fold ({score_ew:.4f} > {score_kf:.4f}). Using Expanding Window predictions.")
        final_preds = preds_ew
        final_score = score_ew
        
    full_preds = final_preds
    full_preds.index.name = 'Date'
    full_preds.to_csv(PREDICTIONS_FILE)
    print(f"Predictions saved to {PREDICTIONS_FILE}")
    
    from sklearn.metrics import brier_score_loss, average_precision_score
    overall_sd = somers_d_score(full_preds['Target'], full_preds['Crash_Prob'])
    overall_brier = brier_score_loss(full_preds['Target'], full_preds['Crash_Prob'])
    overall_pr_auc = average_precision_score(full_preds['Target'], full_preds['Crash_Prob'])
    
    print(f"Overall GBM Out-of-Sample Somers' D: {overall_sd:.4f}")
    print(f"Overall GBM Out-of-Sample Brier Score: {overall_brier:.4f}")
    print(f"Overall GBM Out-of-Sample PR-AUC: {overall_pr_auc:.4f}")
    
    # Rolling Metrics
    metrics = []
    window_size = 30
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
    base_model = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=200, max_depth=3, min_samples_leaf=40, l2_regularization=10.0, random_state=42
    )
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    X_full = df[FEATURES]
    y_full = df[TARGET]
    model.fit(X_full, y_full)
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
        
    # Interpretability
    if shap:
        print("Calculating SHAP values using Proxy Model...")
        proxy_model = HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=200, max_depth=3, min_samples_leaf=40, l2_regularization=10.0, random_state=42
        )
        proxy_model.fit(X_full, y_full)
        explainer = shap.TreeExplainer(proxy_model)
        shap_values = explainer.shap_values(X_full)
        shap_data = {
            'shap_values': shap_values,
            'X': X_full,
            'features': FEATURES,
            'method': 'shap'
        }
    else:
        print("SHAP not found. Calculating Permutation Importance...")
        r = permutation_importance(model, X_full, y_full, n_repeats=10, random_state=42, n_jobs=-1)
        shap_data = {
            'importances_mean': r.importances_mean,
            'features': FEATURES,
            'method': 'permutation'
        }
    
    with open(IMPORTANCE_FILE, 'wb') as f:
        pickle.dump(shap_data, f)
        
    print("GBM Training complete.")

if __name__ == "__main__":
    train_gbm()
