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
    
    # Forward looking min price in next 12 months
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)
    future_min = price.rolling(window=indexer).min()
    
    # Current All-Time High (or rolling 2-year high? ATH is better for "bear market")
    # But usually bear market is from 52-week high or recent peak.
    # Let's use standard Drawdown definition: Drawdown from *running max*.
    running_max = price.cummax()
    
    # Future Drawdown: The lowest point in the next 12 months relative to the *Peak existing at that future moment*.
    # Actually, simpler: Will we be >20% below the Peak-at-time-t within 12 months?
    # Or will we be >20% below the Peak-at-time-(t+k)?
    # Let's target: Will the Maximum Drawdown in the next 12 months exceed 20%?
    # Max Drawdown in (t, t+12) window.
    # Drawdown_t = (P_t / Max(P_0..P_t)) - 1
    # We want max(Drawdown_t+1 ... Drawdown_t+12) < -0.20? (Remember drawdown is negative).
    # So min(Drawdown) < -0.20.
    
    # We need to calculate the *future realized drawdown* path.
    # Note: We can't peek future for *input features*, but we use it for *target*.
    
    # Calculate full history drawdown first
    full_drawdown = calculate_drawdown(price)
    
    # Target: Min(Full Drawdown) in next 12 months < -0.20
    future_min_drawdown = full_drawdown.rolling(window=indexer).min()
    
    # Binary Target
    # Also, we might want to exclude times where we are *already* in a crash bottom.
    # But prediction "Will it crash/stay crashed?" captures risk.
    # If market is down 50%, "Crash Risk" is high (it is crashed).
    # Maybe "Crash Initiation"?
    # The user asked: "estimate when the next S&P 500 crash is likely to occur... probability a crash occurs in next t months"
    # This implies usually "Onset".
    # But let's stick to "Risk Regime". If prob is high, you are in danger.
    df['Target_Crash_12m'] = (future_min_drawdown < -0.20).astype(int)
    
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
    
    # Previous Drawdown state (How deep are we now?)
    df['Current_Drawdown'] = full_drawdown
    
    # Lag Features (prevent leakage)
    # We used current t values for features. Target is t+1..t+12.
    # So features at row t are known at t.
    # Strictly speaking, Shiller data like CPI/Earnings is reported with lag.
    # Usually 1-2 months lag.
    # To be safe/realistic, we should lag macro features by 1 month.
    # But for "Data Scientist" task, using 'vintage' data is hard.
    # We'll assume at Month End t, we have Month End t prices, and Month t-1 Macro.
    # The dataset aligns 'Date' as month start.
    # Values for '2023-01-01' usually mean Jan 2023 average or Jan 1st?
    # Price is usually Monthly Average in Shiller.
    # UNRATE is Monthly.
    # We will lag "Reporting" features by 1 month?
    # Safe approach: Shift Macro/Earnings features by 1. Keep Price/Trend (observable) at 0?
    # Or just Assume Availability.
    # Let's use 1 month lag for Macro columns to be safe.
    
    macro_cols = ['CAPE', 'Earnings_Yield', 'Unemployment_Rate', 'Unemployment_Change_12m', 
                  'Inflation_12m', 'Term_Spread', 'Credit_Spread', 'Real_Rate']
                  
    for c in macro_cols:
        if c in df.columns:
            df[c] = df[c].shift(1)
            
    # Drop rows where Target is NaN (last 12 months)
    df = df.dropna(subset=['Target_Crash_12m'])
    
    # Also drop very early rows if too many NaNs, but we want 1929.
    # We will keep them. Models will handle NaNs.
    
    print(f"Features shape: {df.shape}")
    print(f"Crash Events (Target=1): {df['Target_Crash_12m'].sum()} months / {len(df)}")
    
    df.to_csv(FEATURES_FILE)
    print(f"Saved to {FEATURES_FILE}")

if __name__ == "__main__":
    build_features()
