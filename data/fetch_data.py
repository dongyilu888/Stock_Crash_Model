import pandas as pd
import pandas_datareader.data as web
import os
import requests
import datetime

# Constants
DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw')
SHILLER_URL = 'http://www.econ.yale.edu/~shiller/data/ie_data.xls'
SHILLER_FILE = os.path.join(RAW_DIR, 'shiller_data.xls')

# FRED Series IDs
# Mnemonic: Series Name (Frequency, Start Year usually)
FRED_SERIES = {
    'UNRATE': 'Unemployment Rate',               # Monthly, 1948
    'INDPRO': 'Industrial Production',           # Monthly, 1919
    'CPIAUCSL': 'CPI All Urban Consumers',       # Monthly, 1947
    'GS10': '10-Year Treasury Constant Maturity',# Monthly, 1953
    'TB3MS': '3-Month Treasury Bill',            # Monthly, 1934
    'AAA': 'Moody\'s Seasoned Aaa Corp Bond',    # Monthly, 1919
    'BAA': 'Moody\'s Seasoned Baa Corp Bond',    # Monthly, 1919
    'PPIACO': 'PPI All Commodities',             # Monthly, 1913 (Proxy for commodities)
    'PCOPPUSDM': 'Global Price of Copper',       # Monthly, 1990
    'POILBREUSDM': 'Global Price of Brent Crude',# Monthly, 1990
}

def setup_dirs():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)

def fetch_shiller_data():
    print("Fetching Shiller data...")
    try:
        # Download file first to save it
        response = requests.get(SHILLER_URL)
        with open(SHILLER_FILE, 'wb') as f:
            f.write(response.content)
        print(f"Saved Shiller data to {SHILLER_FILE}")
    except Exception as e:
        print(f"Error fetching Shiller data: {e}")

def fetch_fred_data():
    print("Fetching FRED data...")
    start_date = datetime.datetime(1919, 1, 1)
    end_date = datetime.datetime.now()
    
    for series_id, name in FRED_SERIES.items():
        try:
            print(f"Downloading {name} ({series_id})...")
            df = web.DataReader(series_id, 'fred', start_date, end_date)
            output_path = os.path.join(RAW_DIR, f"{series_id}.csv")
            df.to_csv(output_path)
            print(f"Saved {series_id} to {output_path}")
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")

if __name__ == "__main__":
    setup_dirs()
    fetch_shiller_data()
    fetch_fred_data()
