import yfinance as yf
import pandas as pd
from datetime import date, timedelta

today = date.today()
start = date(today.year-1, today.month, today.day)
end = today

tickers = ['^VIX', '^VIX3M', 'RSP', 'HYG', 'LQD']

print(f"Testing yfinance download for: {tickers}")
print(f"Date range: {start} to {end}")

for t in tickers:
    print(f"\n--- Fetching {t} ---")
    try:
        df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            print(f"FAILED: {t} returned empty DataFrame.")
        else:
            print(f"SUCCESS: {t} returned {len(df)} rows.")
            print(f"Columns: {df.columns.tolist()}")
            # Check for price column
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join([str(c) for c in col]).strip('_') for col in df.columns]
            
            candidates = [c for c in df.columns if any(k in str(c).lower() for k in ['price','close','adj close','adj_close'])]
            print(f"Price candidates found: {candidates}")
            
    except Exception as e:
        print(f"ERROR: {t} raised exception: {e}")
