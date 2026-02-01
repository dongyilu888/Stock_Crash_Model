import pandas as pd
import numpy as np
import os

# Constants
DATA_DIR = 'data'
PROCESSED_FILE = os.path.join(DATA_DIR, 'processed.csv')
FEATURES_FILE = os.path.join(DATA_DIR, 'features.csv')

def calculate_drawdown(prices):
    """
    Calculate drawdown from running maximum.
    """
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    return drawdown

def build_features():
    print("Building features...")
    # Load with index_col=0 (first column) and parse that as dates
    df = pd.read_csv(PROCESSED_FILE, index_col=0, parse_dates=[0])
    df.index.name = 'Date' # Restore name explicitly
    print(f"Initial load shape: {df.shape}")

    
    # 1. Target Definition: Crash in Next 12 Months
    # Definition: "A crash occurs when the S&P 500 experiences a greater than 20% peak-to-trough decline"
    # We want to predict if such an event starts or is imminent.
    # Approach:
    # Look forward 12 months.
    # Calculate the Minimum Price in [t+1, t+12].
    # Calculate the Reference Peak at time t (Max Price in [t-24, t]). (Or All-time High?)
    # People usually define crash from ATH or Local High. 
    # Let's use All-Time High up to time t.
    
    price = df['S&P500_Price']
    full_drawdown = calculate_drawdown(price)
    
    # Target: Min(Full Drawdown) in next 12 months < -0.20
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)
    future_min_drawdown = full_drawdown.rolling(window=indexer).min()
    df['Target_Crash_12m'] = (future_min_drawdown < -0.20).astype(int)
    
    # 1b. Rigorous Survival Target: Bull Market Subjects
    # - Event: A peak that leads to a >20% drawdown.
    # - Start (Clock Reset): The trough before the bull run.
    # - Left-Truncation: Handle subjects that were already "alive" at study start.
    
    # Step 1: Identify all "Crash Decided" months (when price is 20% below recent peak)
    crash_regime = full_drawdown < -0.20
    
    # Step 2: Identify Cycle Landmarks (Robust Peak/Trough Detection)
    # 20% Threshold for Bull/Bear state machine
    peaks = []
    troughs = []
    
    current_high = price.iloc[0]
    current_low = price.iloc[0]
    high_date = price.index[0]
    low_date = price.index[0]
    
    state = "BULL"
    for date, p in price.items():
        if state == "BULL":
            if p > current_high:
                current_high = p
                high_date = date
            elif p < current_high * 0.80:
                peaks.append(high_date)
                state = "BEAR"
                current_low = p
                low_date = date
        else: # BEAR
            if p < current_low:
                current_low = p
                low_date = date
            elif p > current_low * 1.20:
                troughs.append(low_date)
                state = "BULL"
                current_high = p
                high_date = date
                
    peaks = sorted(list(set(peaks)))
    troughs = sorted(list(set(troughs)))
    
    # Step 3: Assign each month to a "Subject" (Bull Market Cycle)
    df['Subject_Start'] = pd.NaT
    df['Event_Observed'] = 0
    df['Death_Date'] = pd.NaT
    
    # Handle the very first bull run
    current_trough = pd.Timestamp('1921-08-01') # Historical floor
    
    for p_date in peaks:
        mask = (df.index > current_trough) & (df.index <= p_date)
        df.loc[mask, 'Subject_Start'] = current_trough
        df.loc[df.index == p_date, 'Event_Observed'] = 1
        df.loc[mask, 'Death_Date'] = p_date
        
        # Find the trough that followed this peak
        post_peak_troughs = [t for t in troughs if t > p_date]
        if post_peak_troughs:
            current_trough = post_peak_troughs[0]
            
    # Step 3b: Identify Contraction Phases (Peak to Trough) for Visualization
    df['Contraction_Phase'] = 0
    for p_date in peaks:
        post_peak_troughs = [t for t in troughs if t > p_date]
        if post_peak_troughs:
            t_date = post_peak_troughs[0]
            mask_crash = (df.index > p_date) & (df.index <= t_date)
            df.loc[mask_crash, 'Contraction_Phase'] = 1





    
    # Handle the final (ongoing) cycle
    mask_ongoing = df['Subject_Start'].isna() & (df.index > current_trough)
    df.loc[mask_ongoing, 'Subject_Start'] = current_trough
    df.loc[mask_ongoing, 'Death_Date'] = df.index.max()
    df.loc[mask_ongoing, 'Event_Observed'] = 0 # Censored
    
    # Step 4: Calculate Durations and Entry Times (Left-Truncation)
    def months_diff(d1, d2):
        if pd.isna(d1) or pd.isna(d2): return 0
        return (d1.year - d2.year) * 12 + (d1.month - d2.month)
    
    # Current Age of the bull market
    df['Current_Age'] = df.apply(lambda row: months_diff(row.name, row['Subject_Start']), axis=1)
    
    # Final age when bull market dies/censored
    df['Duration_to_Crash'] = df.apply(lambda row: months_diff(row['Death_Date'], row['Subject_Start']), axis=1)

    # Entry Age for this specific observation
    df['Entry_Age'] = df['Current_Age']
    df['Stop_Age'] = df['Current_Age'] + 1
    
    # Event only happens if current month is the peak
    df['Survival_Event'] = df['Event_Observed'] 
    
    # Note: We do not dropna(subset=['Subject_Start']) here to allow
    # dashboarding contraction months.

    
    # Ensure all ages are non-negative
    df = df[df['Current_Age'] >= 0]
    
    # 2. Features
    
    # Valuation
    # CAPE (Should exist from Shiller) - Fill forward if missing recent months (Shiller lags)
    df['CAPE'] = df['CAPE'].ffill()
    
    # Earnings Yield (E/P) using Shiller data
    # Earnings usually trail 10 years for CAPE, or TTM for PE.
    # Shiller 'Earnings' column is usually TTM or similar.
    df['Earnings_Yield'] = df['Earnings'] / df['S&P500_Price']
    
    # Macro
    # Unemployment: Change YoY
    # FRED: UNRATE.
    if 'UNRATE' in df.columns:
        df['Unemployment_Rate'] = df['UNRATE']
        df['Unemployment_Change_12m'] = df['UNRATE'].diff(12)
        df['Unemployment_Change_3m'] = df['UNRATE'].diff(3)
    else:
        # Create dummy if missing (e.g. 1929 era?)
        # Shiller doesn't have unemployment.
        # We might drop 1929 rows if strict, or kept as NaN.
        pass
        
    # Inflation
    # Shiller has CPI.
    # CPI YoY
    df['Inflation_12m'] = df['CPI_Shiller'].pct_change(12)
    
    # Rates/Credit
    # Term Spread: GS10 - TB3MS. (Use Shiller Long Rate if GS10 missing)
    # Shiller 'Interest_Rate_Long_Shiller' is available 1871+.
    # We can use Shiller rate as GS10 proxy if GS10 is null.
    df['Long_Rate'] = df['GS10'].fillna(df['Interest_Rate_Long_Shiller'])
    
    # Short Rate: TB3MS from FRED. Before 1934? 
    # Shiller doesn't have short rate easily.
    # We accept NaN before 1934 for Term Spread? Or use constant? 
    # Let's leave NaN.
    if 'TB3MS' in df.columns:
        df['Term_Spread'] = df['Long_Rate'] - df['TB3MS']
    
    # Credit Spreads: BAA - AAA.
    # FRED has these long history (1919+).
    if 'BAA' in df.columns and 'AAA' in df.columns:
        df['Credit_Spread'] = df['BAA'] - df['AAA']
        
    # Real Rate
    df['Real_Rate'] = df['Long_Rate'] - (df['Inflation_12m'] * 100) # Inflation is fractional? pct_change gives fraction. Rate is percent (e.g. 5.0).
    # Adjust: pct_change * 100
    df['Real_Rate'] = df['Long_Rate'] - (df['Inflation_12m'] * 100)
    
    # Commodities
    # Proxy: Gold, Oil.
    # Variable availability.
    # Use 12m Returns.
    # Handle NaNs: Fill with 0 or exclude.
    if 'PPI_Commodity' in df.columns:
        df['Commodity_Ret_12m'] = df['PPI_Commodity'].pct_change(12)
        
    # Market Internals (Momentum, Vol)
    df['S&P_Ret_12m'] = df['S&P500_Price'].pct_change(12)
    df['S&P_Vol_12m'] = df['S&P500_Price'].rolling(12).std() / df['S&P500_Price'] # Simple monthly vol proxy
    
    # Leading Indicators
    if 'Housing_Starts' in df.columns:
        df['Housing_Starts_12m'] = df['Housing_Starts'].pct_change(12)
    if 'Building_Permits' in df.columns:
        df['Building_Permits_12m'] = df['Building_Permits'].pct_change(12)
    if 'Consumer_Sentiment' in df.columns:
        df['Sentiment_Change_12m'] = df['Consumer_Sentiment'].diff(12)
    if 'Weekly_Hours_Mfg' in df.columns:
        df['Mfg_Hours_Change_12m'] = df['Weekly_Hours_Mfg'].diff(12)

    # Interaction & Non-linear Features
    # 1. Stress * Valuation interaction (high unemployment change + high CAPE)
    if 'CAPE' in df.columns and 'Unemployment_Change_12m' in df.columns:
        df['CAPE_Unemp_Interaction'] = df['CAPE'] * df['Unemployment_Change_12m'].clip(lower=0)
        
    # 2. Term Inversion (Inverse of spread, focusing on the inversion regime)
    if 'Term_Spread' in df.columns:
        df['Term_Inversion'] = 1.0 / (df['Term_Spread'] + 2.0) # Offset to avoid div by zero and focus on inversion
        
    # Previous Drawdown state (How deep are we now?)
    df['Current_Drawdown'] = full_drawdown
    
    # Feature-Specific Lags: Apply realistic publication delays
    # This optimizes model responsiveness while preventing data leakage
    lag_config = {
        # Market data: available immediately (end-of-day)
        'S&P_Ret_12m': 0,
        'S&P_Vol_12m': 0,
        'Current_Drawdown': 0,
        
        # Interest rates: available within days (Fed/Treasury data)
        'Term_Spread': 0,
        'Credit_Spread': 0,
        'Real_Rate': 0,
        'Term_Inversion': 0,
        
        # Valuation: quarterly earnings with ~1 month processing
        'CAPE': 1,
        'Earnings_Yield': 1,
        
        # Macro: monthly releases with 2-4 week delay
        'Unemployment_Rate': 1,
        'Unemployment_Change_12m': 1,
        'Unemployment_Change_3m': 1,
        'Inflation_12m': 1,  # CPI released mid-month for prior month
        'Commodity_Ret_12m': 1,
        'Mfg_Hours_Change_12m': 1,
        
        # Housing: 1-2 month lag (Census Bureau releases)
        'Housing_Starts_12m': 2,
        'Building_Permits_12m': 2,
        
        # Sentiment: end of month release
        'Sentiment_Change_12m': 1,
        
        # Derived features inherit from components
        'CAPE_Unemp_Interaction': 1,  # Max of CAPE(1) and Unemployment(1)
    }
    
    # Apply feature-specific lags
    for feature, lag_months in lag_config.items():
        if feature in df.columns:
            df[feature] = df[feature].shift(lag_months)
            
    # We DO NOT dropna for Target here to allow dashboarding current months
    # Retraining scripts will handle their own dropna.


    
    # Also drop very early rows if too many NaNs, but we want 1929.
    # We will keep them. Models will handle NaNs.
    
    print(f"Features shape: {df.shape}")
    print(f"Crash Events (Target=1): {df['Target_Crash_12m'].sum()} months / {len(df)}")
    
    df.to_csv(FEATURES_FILE)
    print(f"Saved to {FEATURES_FILE}")

if __name__ == "__main__":
    build_features()
