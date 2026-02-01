import pandas as pd
import numpy as np
import os
import datetime

# Constants
DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_FILE = os.path.join(DATA_DIR, 'processed.csv')
SHILLER_FILE = os.path.join(RAW_DIR, 'shiller_data.xls')

FRED_FILES = {
    'UNRATE.csv': 'UNRATE',
    'INDPRO.csv': 'INDPRO',
    'CPIAUCSL.csv': 'CPI_FRED',
    'GS10.csv': 'GS10',
    'TB3MS.csv': 'TB3MS',
    'AAA.csv': 'AAA',
    'BAA.csv': 'BAA',
    'PPIACO.csv': 'PPI_Commodity',
    'PCOPPUSDM.csv': 'Copper',
    'POILBREUSDM.csv': 'Oil_Brent',
    'HOUST.csv': 'Housing_Starts',
    'UMCSENT.csv': 'Consumer_Sentiment',
    'AWHMAN.csv': 'Weekly_Hours_Mfg',
    'PERMIT.csv': 'Building_Permits'
}

def load_shiller():
    print("Processing Shiller data...")
    try:
        # Load all sheets to find the right one
        xls = pd.ExcelFile(SHILLER_FILE)
        print(f"Sheets found: {xls.sheet_names}")
        
        # Prefer 'Data' sheet if exists
        sheet_name = 'Data' if 'Data' in xls.sheet_names else xls.sheet_names[0]
        
        # Read header section
        df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=20)
        
        # Find row with 'Date' AND 'P' or 'Price' or 'SP500' in the same row
        header_idx = -1
        for i, row in df_raw.iterrows():
            row_str = row.astype(str).str.lower()
            # Check for 'date' and ('p' or 'price')
            # Shiller header usually: 'Date', 'P', 'D', 'E', 'CPI'...
            # 'p' is short, so check strict 'p' or 'price'.
            # Note row_str is a series of strings.
            has_date = row_str.str.contains('date').any()
            has_price = row_str.eq('p').any() or row_str.str.contains('price').any() or row_str.str.contains('s&p').any()
            
            if has_date and has_price:
                header_idx = i
                break
        
        if header_idx == -1:
            # Fallback: Search for row starting with 'Date' exact match?
            # Or just default to 7.
            print("Could not find robust header. Trying to find any 'Date'.")
            for i, row in df_raw.iterrows():
                 if row.astype(str).str.contains('date', case=False).any():
                     header_idx = i
                     break
        
        if header_idx == -1:
             print("Still failed. Defaulting to 7.")
             header_idx = 7
             
        print(f"Using header row: {header_idx}")
        
        # Read full file
        df = pd.read_excel(xls, sheet_name=sheet_name, header=header_idx)
        print(f"Columns found: {df.columns.tolist()}")
        
        # Use first column as Date
        df = df.rename(columns={df.columns[0]: 'Date'})
        
        # Check type
        print(f"Date column type: {type(df['Date'])}")
        
        # Remove duplicate columns if any (keep first)
        df = df.loc[:, ~df.columns.duplicated()]

        # Function to parse Shiller date (YYYY.MM where .1 = Oct?? or .10 = Oct?)
        print(f"First few date values: {df['Date'].head().tolist()}")
        
        def parse_single_date(d):
            try:
                d_num = pd.to_numeric(d, errors='coerce')
                if pd.isna(d_num):
                    return np.nan
                    
                year = int(d_num)
                # 2023.01 -> 0.01. 0.01*100 = 1.
                # 2023.10 -> 0.1. 0.1*100 = 10.
                frac = d_num - year
                month = int(round(frac * 100))
                
                if month < 1 or month > 12:
                    return np.nan
                    
                return pd.Timestamp(year=year, month=month, day=1)
            except:
                return np.nan

        df['Date_Parsed'] = df['Date'].apply(parse_single_date)
        df = df.dropna(subset=['Date_Parsed'])
        
        # Verify
        if df.empty:
            print("Shiller dataframe empty after date parsing.")
            return pd.DataFrame()
            
        df = df.set_index('Date_Parsed').sort_index()
        df.index.name = 'Date'
        
        # Select relevant columns and rename
        col_map = {
            'P': 'S&P500_Price',
            'D': 'Dividends',
            'E': 'Earnings',
            'CPI': 'CPI_Shiller',
            'Rate GS10': 'Interest_Rate_Long_Shiller',
            'Rate': 'Interest_Rate_Long_Shiller',
            'Long Interest Rate': 'Interest_Rate_Long_Shiller',
            'CAPE': 'CAPE'
        }
        
        rename_dict = {}
        for col in df.columns:
            c_str = str(col).strip()
            # Direct match
            if c_str in col_map:
                rename_dict[col] = col_map[c_str]
                continue
                
            # Partial match for Rate and CAPE
            if 'CAPE' in c_str:
                rename_dict[col] = 'CAPE'
            elif 'Rate' in c_str and 'Long' in c_str:
                 rename_dict[col] = 'Interest_Rate_Long_Shiller'
            elif c_str == 'P': # Strict P
                rename_dict[col] = 'S&P500_Price'
        
        df = df.rename(columns=rename_dict)
        
        # Remove duplicates after rename (e.g. Rate and Rate GS10 both might map to same if logic was loose)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Keep only known columns
        cols_to_keep = [c for c in ['S&P500_Price', 'Dividends', 'Earnings', 'CPI_Shiller', 'Interest_Rate_Long_Shiller', 'CAPE'] if c in df.columns]
        
        # Ensure numeric
        for c in cols_to_keep:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        return df[cols_to_keep]
        
    except Exception as e:
        print(f"Error processing Shiller: {e}")
        # Return empty DF to allow pipeline to proceed with FRED only if needed (though not ideal)
        return pd.DataFrame()

def load_fred():
    print("Processing FRED data...")
    dfs = []
    for f, name in FRED_FILES.items():
        path = os.path.join(RAW_DIR, f)
        if os.path.exists(path):
            try:
                d = pd.read_csv(path, parse_dates=['DATE'], index_col='DATE')
                d = d.rename(columns={d.columns[0]: name})
                # Resample to Month Start just in case (FRED usually MS)
                d = d.resample('MS').first()
                dfs.append(d)
            except Exception as e:
                print(f"Error reading {f}: {e}")
    
    if dfs:
        return pd.concat(dfs, axis=1)
    return pd.DataFrame()

def main():
    shiller_df = load_shiller()
    fred_df = load_fred()
    
    # Merge
    # Outer join to keep recent data if Shiller is lagging, or vice versa
    # But usually we align to Month Start
    
    # Ensure indices are Month Start
    if not shiller_df.empty:
        shiller_df = shiller_df.resample('MS').first()
        
    merged = pd.merge(shiller_df, fred_df, left_index=True, right_index=True, how='outer')
    
    # Sort
    merged = merged.sort_index()
    
    # Filter 1920 onwards for relevance (user asking for 1929, but slightly earlier ok)
    merged = merged['1920-01-01':]
    
    # Forward fill logic?
    # Some macro data is quarterly (like GDP which we don't have, but others might be).
    # Shiller data is monthly.
    # FRED is monthly.
    # We shouldn't need fill except for alignment.
    
    print(f"Merged shape: {merged.shape}")
    print(f"Saving to {PROCESSED_FILE}")
    merged.to_csv(PROCESSED_FILE)

if __name__ == "__main__":
    main()
