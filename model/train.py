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
    'Commodity_Ret_12m', 'S&P_Ret_12m', 'S&P_Vol_12m'
]

TARGET = 'Target_Crash_12m'

def somers_d_score(y_true, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
        return 2 * auc - 1
    except:
        return np.nan

def train_model():
    print("Training model with expanding window...")
    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=[0])
    df = df.sort_index()
    
    # Define start date for backtest (e.g. 1960 to have enough history)
    start_date = pd.Timestamp('1960-01-01')
    
    # Expanding Window Loop
    # We will iterate year by year? Or month by month?
    # Year by year is faster and sufficient for "Risk regimes".
    # We predict for the *next* year (or just store predictions for that test set).
    # Actually, we can predict for every month in the "test" year.
    
    predictions = []
    
    # Create timestamps for annual expansion
    # Start from 1960
    years = pd.date_range(start=start_date, end=df.index.max(), freq='AS') # Year Start
    
    # Model Pipeline
    # Impute missing values (RF can't handle NaNs in scikit-learn standard, though HistGradient can. use SimpleImputer)
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
    ])
    
    running_metrics = []
    
    # We need at least some data before start_date
    if df.index.min() > start_date:
        print(f"Warning: Data starts at {df.index.min()}, adjusting start.")
        years = pd.date_range(start=df.index.min() + pd.DateOffset(years=5), end=df.index.max(), freq='AS')
    
    print(f"Backtesting from {years[0].year} to {years[-1].year}...")
    
    for dt in years:
        train_mask = df.index < dt
        test_mask = (df.index >= dt) & (df.index < dt + pd.DateOffset(years=1))
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if test_df.empty:
            continue
            
        if len(train_df) < 50: # Need minimum samples
            continue
            
        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]
        X_test = test_df[FEATURES]
        # y_test is not needed for prediction, but for eval later
        
        # Fit
        model.fit(X_train, y_train)
        
        # Predict Probabilities
        probs = model.predict_proba(X_test)[:, 1]
        
        # Store
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
    
    # Calculate Evaluation Scores
    # Overall Somers' D
    overall_sd = somers_d_score(full_preds['Target'], full_preds['Crash_Prob'])
    print(f"Overall Out-of-Sample Somers' D: {overall_sd:.4f}")
    
    # Rolling Somers' D (e.g. 5 or 10 year window)
    # We can compute this for the metrics file
    metrics = []
    # Compute score per decade? Or rolling 5 year?
    # Window size: 60 months
    window_size = 60
    
    # We need a rolling apply.
    # Custom rolling metric.
    # Note: Rolling AUC requires both classes present in window.
    
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
    X_full = df[FEATURES]
    y_full = df[TARGET]
    model.fit(X_full, y_full)
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
        
    # SHAP Explainer
    try:
        import shap
        # Use TreeExplainer (but we have a Pipeline). 
        # Extract step.
        rf_model = model.named_steps['classifier']
        
        # SHAP needs non-missing data. Valid set.
        # We need to transform X_full first.
        imputer = model.named_steps['imputer']
        X_transformed = pd.DataFrame(imputer.transform(X_full), columns=FEATURES, index=X_full.index)
        
        # Explain
        # Using a subsample for background if dataset is large, but 1000 rows is fine.
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_transformed)
        
        # shap_values is list of arrays for classification (one per class). We want Prob(Crash)=1
        if isinstance(shap_values, list):
             # Class 1
             vals = shap_values[1]
        else:
             vals = shap_values
             
        # Save SHAP values (associated with indices)
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
        # Fallback: Save simple feature importances
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
