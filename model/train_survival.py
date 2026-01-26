import pandas as pd
import numpy as np
import os
import pickle
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Constants
DATA_DIR = 'data'
FEATURES_FILE = os.path.join(DATA_DIR, 'features.csv')
RESULTS_DIR = 'results'
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'predictions_survival.csv')
COEFFICIENTS_FILE = os.path.join(RESULTS_DIR, 'coefficients_survival.csv')
MODEL_FILE = os.path.join(RESULTS_DIR, 'latest_model_survival.pkl')

FEATURES = [
    'CAPE', 'Earnings_Yield', 
    'Inflation_12m', 'Unemployment_Rate', 'Unemployment_Change_12m', 
    'Term_Spread', 'Credit_Spread', 'Real_Rate', 
    'Commodity_Ret_12m', 'S&P_Ret_12m', 'S&P_Vol_12m'
]

DURATION = 'Duration_to_Crash'
EVENT = 'Event_Observed'

def train_survival():
    print("Training Survival Model (CoxPH)...")
    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=[0])
    df = df.sort_index()
    
    # Pre-process: Drop rows with NaNs in features/target
    # CoxPH doesn't handle NaNs natively in many implementations
    model_df = df[FEATURES + [DURATION, EVENT]].dropna()
    
    # Expanding Window Backtest
    start_date = pd.Timestamp('1960-01-01')
    predictions = []
    years = pd.date_range(start=start_date, end=df.index.max(), freq='YS')
    
    c_indices = []
    
    cph = CoxPHFitter(penalizer=0.1) # Add some regularization
    
    print(f"Backtesting from {years[0].year} to {years[-1].year}...")
    
    for dt in years:
        train_df = model_df[model_df.index < dt]
        test_df = model_df[(model_df.index >= dt) & (model_df.index < dt + pd.DateOffset(years=1))]
        
        if test_df.empty or len(train_df) < 100:
            continue
            
        # Fit on past
        try:
            cph.fit(train_df, duration_col=DURATION, event_col=EVENT)
            
            # Predict partial hazard (Risk Score)
            hazard_score = cph.predict_partial_hazard(test_df[FEATURES])
            
            res = pd.DataFrame({
                'Hazard_Score': hazard_score,
                'Duration': test_df[DURATION],
                'Event': test_df[EVENT]
            }, index=test_df.index)
            predictions.append(res)
            
            # C-index for this window
            c_index = concordance_index(test_df[DURATION], -hazard_score, test_df[EVENT])
            c_indices.append({'Date': dt, 'C_Index': c_index})
        except:
            continue
            
    if not predictions:
        print("No predictions generated.")
    else:
        full_preds = pd.concat(predictions)
        full_preds.index.name = 'Date'
        full_preds.to_csv(PREDICTIONS_FILE)
        
        metrics_df = pd.DataFrame(c_indices).set_index('Date')
        metrics_df.to_csv(os.path.join(RESULTS_DIR, 'metrics_survival.csv'))
        
        overall_c = concordance_index(full_preds['Duration'], -full_preds['Hazard_Score'], full_preds['Event'])
        print(f"Overall Survival C-Index: {overall_c:.4f}")
    
    # Final Model on full history
    print("Training final survival model...")
    cph.fit(model_df, duration_col=DURATION, event_col=EVENT)
    
    # Save Model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(cph, f)
        
    # Coefficients for Interpretation
    cph.summary.to_csv(COEFFICIENTS_FILE)
    
    # Survival Curve for TODAY
    # Take latest known features
    latest_X = model_df[FEATURES].iloc[[-1]]
    survival_curve = cph.predict_survival_function(latest_X)
    survival_curve.to_csv(os.path.join(RESULTS_DIR, 'current_survival_curve.csv'))
    
    print("Survival training complete.")

if __name__ == "__main__":
    train_survival()
