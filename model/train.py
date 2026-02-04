import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Constants
DATA_DIR = 'data'
FEATURES_FILE = os.path.join(DATA_DIR, 'features.csv')
RESULTS_DIR = 'results'
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'predictions.csv')
METRICS_FILE = os.path.join(RESULTS_DIR, 'metrics.csv')
MODEL_FILE = os.path.join(RESULTS_DIR, 'latest_model.pkl')
SHAP_FILE = os.path.join(RESULTS_DIR, 'shap_values.pkl')

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

def run_expanding_window(df):
    print(f"Running Expanding Window Validation (Start 1960)...")
    start_date = pd.Timestamp('1929-01-01')
    years = pd.date_range(start=pd.Timestamp('1960-01-01'), end=df.index.max(), freq='YS')
    
    if df.index.min() > start_date:
        years = pd.date_range(start=df.index.min() + pd.DateOffset(years=5), end=df.index.max(), freq='AS')
        
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
    ])
    
    predictions = []
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

def run_purged_kfold(df, n_splits=3):
    print(f"Running Purged K-Fold CV (K={n_splits})...")
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    predictions = []
    
    # KFold indices are integer based on len(df)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        # Identify Test Dates
        test_dates = df.index[test_idx]
        test_start = test_dates.min()
        test_end = test_dates.max()
        
        # Purge Train Data
        # Drop training samples where label might overlap with Test period labels
        # Embargo: 12 months before Test Start AND 12 months after Test End
        # If test set is 1960-1980.
        # Train data 1959-1960 (Label 12m forward) -> overlaps with Test 1960.
        # Train data 1980-1981 (Label 12m forward) -> overlaps with Test 1980? No.
        # But a training sample at 1980 (looking 12m forward) uses the same future as Test sample at 1980.
        # The standard strict rule: No overlap in (Observation Time + 12m Target Window).
        
        # Define Forbidden Zone for Training OBSERVATION times.
        # Training observation at T has target [T, T+12].
        # Test observation at t has target [t, t+12].
        # We need [T, T+12] and [t, t+12] to NOT overlap.
        # So T must be < t - 12 months.
        # OR T must be > t + 12 months.
        
        train_original_idx = df.index[train_idx]
        
        mask_before = train_original_idx < (test_start - pd.DateOffset(months=12))
        mask_after = train_original_idx > (test_end + pd.DateOffset(months=12))
        
        train_purged_dates = train_original_idx[mask_before | mask_after]
        
        if len(train_purged_dates) < 50:
            print(f"Fold {fold_idx}: Not enough training data after purging.")
            continue
            
        train_df = df.loc[train_purged_dates]
        test_df = df.iloc[test_idx]
        
        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]
        X_test = test_df[FEATURES]
        
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
        ])
        
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        
        res = pd.DataFrame({'Crash_Prob': probs, 'Target': test_df[TARGET]}, index=test_df.index)
        predictions.append(res)
        
    if not predictions:
        return pd.DataFrame(), 0.0
        
    full_preds = pd.concat(predictions).sort_index()
    score = somers_d_score(full_preds['Target'], full_preds['Crash_Prob'])
    return full_preds, score

def train_model():
    print("Loading data...")
    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=[0])
    df = df.sort_index()
    
    # 1. Run Expanding Window
    preds_ew, score_ew = run_expanding_window(df)
    print(f"Expanding Window Somers' D: {score_ew:.4f}")
    
    # 2. Run Purged K-Fold
    preds_kf, score_kf = run_purged_kfold(df, n_splits=3)
    print(f"Purged K-Fold (K=3) Somers' D: {score_kf:.4f}")
    
    # Compare and Select
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
    
    # Generate Rolling Metrics from Final Preds
    # Rolling Somers' D (e.g. 5 or 10 year window)
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
    
    # Train final model on ALL data for "Today" view and Interpretation
    print("Training final model on full history...")
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
    ])
    X_full = df[FEATURES]
    y_full = df[TARGET]
    model.fit(X_full, y_full)
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
        
    # SHAP Explainer
    try:
        import shap
        rf_model = model.named_steps['classifier']
        imputer = model.named_steps['imputer']
        X_transformed = pd.DataFrame(imputer.transform(X_full), columns=FEATURES, index=X_full.index)
        
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_transformed)
        
        if isinstance(shap_values, list):
             vals = shap_values[1]
        elif len(shap_values.shape) == 3:
             vals = shap_values[:, :, 1]
        else:
             vals = shap_values
             
        shap_data = {
            'shap_values': vals,
            'X': X_transformed,
            'explainer': explainer,
            'method': 'shap'
        }
        with open(SHAP_FILE, 'wb') as f:
            pickle.dump(shap_data, f)
        print("SHAP values computed and saved.")
        
    except Exception as e:
        print(f"SHAP computation failed: {e}. Falling back to Feature Importance.")
        rf_model = model.named_steps['classifier']
        importances = rf_model.feature_importances_
        shap_data = {
            'feature_importances': importances,
            'features': FEATURES,
            'method': 'importance'
        }
        with open(SHAP_FILE, 'wb') as f:
            pickle.dump(shap_data, f)

    print("Training complete.")

if __name__ == "__main__":
    train_model()
