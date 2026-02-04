import os
import pickle
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.impute import SimpleImputer

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
    'Commodity_Ret_12m', 'S&P_Ret_12m', 'S&P_Vol_12m',
    'Housing_Starts_12m', 'Building_Permits_12m', 'Sentiment_Change_12m', 'Mfg_Hours_Change_12m',
    'CAPE_Unemp_Interaction', 'Term_Inversion'
]

# Snapshot Survival Params
DURATION = 'Stop_Age'
ENTRY = 'Entry_Age'
EVENT = 'Survival_Event'
ID_COL = 'Subject_Start'
TRUE_DURATION = 'Duration_to_Crash'

def train_survival():
    print("Training Survival Model (CoxPH)...")
    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=[0])
    df = df.sort_index()
    
    # Pre-process: Drop rows with NaNs in features/target
    # Snapshot approach: each month is a row with Entry=CurrentAge, Stop=CurrentAge+1
    df['subject_id'] = df[ID_COL].factorize()[0]
    
    # Survival Critical Columns (Must have these to fit/predict)
    survival_cols = ['subject_id', ENTRY, DURATION, EVENT, TRUE_DURATION]
    
    # We ONLY keep FEATURES and survival columns. 
    # Relax strict dropna: only drop if survival target info is missing.
    model_df = df[FEATURES + survival_cols].dropna(subset=survival_cols)
    fit_df = model_df.copy()
    
    # Imputer for missing features (e.g. earlier years)
    imputer = SimpleImputer(strategy='median')
    
    # Expanding Window Backtest
    # Strategy: Start with 30 years of training (1929-1959)
    # Then expand by 3 years each fold (Training: 1929-1960, Test: 1961-1963, etc.)
    start_test_date = pd.Timestamp('1960-01-01')
    predictions = []
    years = pd.date_range(start=start_test_date, end=df.index.max(), freq='3YS')
    
    c_indices = []
    
    cph = CoxPHFitter(penalizer=0.5)
    
    print(f"Backtesting from {years[0].year} to {years[-1].year} with 3-year windows...")
    
    for dt in years:
        # Prepare Training Data
        # At time T, we predict T -> T+3 years. We can train on data up to T-12 months.
        # This aligns with the strict embargo used in other models to verify robustness.
        train_df = fit_df[fit_df.index < (dt - pd.DateOffset(months=12))].copy()
        
        # Test 3 years forward (e.g., 1960, 1961, 1962 inclusive)
        test_df = fit_df[(fit_df.index >= dt) & (fit_df.index < dt + pd.DateOffset(years=3))].copy()
        
        # We need at least 2 years of history to even try
        if test_df.empty or len(train_df) < 24:
            continue
            
        # Impute features in train and test
        # Use local median if available, otherwise 0
        train_medians = train_df[FEATURES].median()
        train_df[FEATURES] = train_df[FEATURES].fillna(train_medians).fillna(0)
        test_df[FEATURES] = test_df[FEATURES].fillna(train_medians).fillna(0)
        
        # Fit on past
        try:
            # Must have at least one event to fit CoxPH
            if train_df[EVENT].sum() == 0:
                continue
                
            # Dynamic Feature Selection: Drop columns with no variance (all constant)
            # This is critical for early years where many features are imputed zeros.
            variances = train_df[FEATURES].var()
            active_features = variances[variances > 1e-9].index.tolist()
            
            if not active_features:
                continue
                
            # We use cluster_col for correlated observations
            # IMPORTANT: Fit ONLY on active_features + target columns
            fit_cols = active_features + [ENTRY, DURATION, EVENT, 'subject_id']
            cph.fit(train_df[fit_cols], duration_col=DURATION, event_col=EVENT, entry_col=ENTRY, cluster_col='subject_id')
            
            # Predict partial hazard (Risk Score) using same active features
            hazard_score = cph.predict_partial_hazard(test_df[active_features])
            
            if TRUE_DURATION not in test_df.columns:
                print(f"MISSING COLUMN at {dt}! Columns: {test_df.columns.tolist()}")
                continue
                
                
            res = pd.DataFrame({
                'Hazard_Score': hazard_score.values,
                'Duration': test_df[TRUE_DURATION].values,
                'Event': test_df[EVENT].values
            }, index=test_df.index)
            
            # Predict Survival Probabilities for IBS (Evaluated at 12, 24, 36 months)
            try:
                surv_funcs = cph.predict_survival_function(test_df[active_features])
                # Get probabilities at specific horizons (closest available time)
                times = [12, 24, 36]
                for t in times:
                    # Find closest time index in survival function
                    idx = min(surv_funcs.index, key=lambda x: abs(x - t))
                    res[f'Surv_Prob_{t}m'] = surv_funcs.loc[idx].values
            except Exception as e:
                print(f"Could not calc survival prob at {dt}: {e}")
                
            predictions.append(res)

            
            # C-index for this window (only if at least one event present)
            if test_df[EVENT].sum() > 0:
                c_index = concordance_index(test_df[TRUE_DURATION].values, -hazard_score.values, test_df[EVENT].values)
                c_indices.append({'Date': dt, 'C_Index': c_index})
        except Exception as e:
            # Only print if it's not a common convergence warning or something trivial
            if "Convergence" not in str(e) and "Singular matrix" not in str(e):
                print(f"Error at {dt}: {e}")
            continue
            
    if not predictions:
        print("No predictions generated.")
        return # Exit early if no results
    else:
        full_preds = pd.concat(predictions)
        full_preds.index.name = 'Date'
        # Scale hazard score for display (it can be very large/small)
        # We keep it as is for consistency with coefficients
        full_preds.to_csv(PREDICTIONS_FILE)
        
        # Calculate Rolling C-Index (60-month window)
        metrics = []
        window_size = 60 # 5 years
        
        for end_idx in range(window_size, len(full_preds)):
            window = full_preds.iloc[end_idx-window_size:end_idx]
            if window['Event'].sum() > 0:
                try:
                    score = concordance_index(window['Duration'], -window['Hazard_Score'], window['Event'])
                except:
                    score = np.nan
            else:
                score = np.nan
            metrics.append({'Date': full_preds.index[end_idx], 'C_Index': score})
            
        if metrics:
            metrics_df = pd.DataFrame(metrics).set_index('Date')
            metrics_df.to_csv(os.path.join(RESULTS_DIR, 'metrics_survival.csv'))
        
        # Overall C-index
        full_preds = full_preds.dropna()
        if full_preds['Event'].sum() > 0:
            overall_c = concordance_index(full_preds['Duration'], -full_preds['Hazard_Score'], full_preds['Event'])
            print(f"Overall Survival C-Index: {overall_c:.4f}")
            
            # Integrated Brier Score (Select Horizons)
            full_preds = full_preds.dropna(subset=['Surv_Prob_12m', 'Surv_Prob_24m', 'Surv_Prob_36m'])
            brier_scores = []
            for t in [12, 24, 36]:
                # Filter useful subjects: 
                # Keep if Event happened before t OR Censored after t.
                # Drop if Censored before t (status unknown at t).
                mask_usable = (full_preds['Duration'] > t) | (full_preds['Event'] == 1)
                subset = full_preds[mask_usable].copy()
                
                if len(subset) > 0:
                    # Target: Did crash happen by time t? (1 if Duration <= t, 0 if Duration > t)
                    subset['Target_at_t'] = (subset['Duration'] <= t).astype(int)
                    # Prob of Event = 1 - Surv_Prob
                    subset['Prob_Event_at_t'] = 1 - subset[f'Surv_Prob_{t}m']
                    
                    bs = ((subset['Target_at_t'] - subset['Prob_Event_at_t']) ** 2).mean()
                    brier_scores.append(bs)
            
            if brier_scores:
                ibs = np.mean(brier_scores)
                print(f"Overall Integrated Brier Score (1-3 Year): {ibs:.4f}")
        else:
            print("No events in backtest, overall C-Index cannot be calculated.")
    
    # Final Model on full history
    print("Training final survival model...")
    # One last imputation on the full set
    fit_df_final = fit_df.copy()
    fit_df_final[FEATURES] = fit_df_final[FEATURES].fillna(fit_df_final[FEATURES].median()).fillna(0)
    
    # Drop zero-variance columns
    final_vars = fit_df_final[FEATURES].var()
    final_active = final_vars[final_vars > 1e-9].index.tolist()
    
    fit_cols = final_active + [ENTRY, DURATION, EVENT, 'subject_id']
    cph.fit(fit_df_final[fit_cols], duration_col=DURATION, event_col=EVENT, entry_col=ENTRY, cluster_col='subject_id')
    
    # Save Model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(cph, f)
        
    # Coefficients for Interpretation
    cph.summary.to_csv(COEFFICIENTS_FILE)
    
    # Survival Curve for TODAY
    try:
        latest_X = fit_df[FEATURES].iloc[[-1]]
        # For prediction, we need to provide the subject's attained age so it predicts *conditional* survival
        # But predict_survival_function often assumes entry=0.
        # We'll just save the coefficients for now.
        pass
    except:
        pass
    
    print("Survival training complete.")

if __name__ == "__main__":
    train_survival()
