import os, pandas as pd, yfinance as yf
from datetime import datetime

def download_spy(start="1993-01-01", end=None, cache_path="data/raw/spy.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if end is None: end = datetime.utcnow().strftime("%Y-%m-%d")
    df = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    if df.empty: raise RuntimeError("yfinance returned empty data for SPY.")
    df = df.reset_index().sort_values("Date").reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df

def load_prices(csv_path="data/raw/spy.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

def add_returns(df: pd.DataFrame, use_adj=True) -> pd.DataFrame:
    price_col = "Adj Close" if use_adj and "Adj Close" in df.columns else "Close"
    out = df.copy()
    out["ret"] = out[price_col].pct_change()
    return out.dropna(subset=["ret"]).reset_index(drop=True)