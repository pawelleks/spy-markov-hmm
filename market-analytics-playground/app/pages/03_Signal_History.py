import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import io
import matplotlib.pyplot as plt

# Minimal dark theme colors (keep consistent)
DARK_BG = "#0b1220"
DARK_AX = "#0e1525"
FG = "#d7e3f3"
LINE = "#9ec4ff"
GREEN = "#4caf50"
RED = "#f44336"
NEUTRAL = "#9e9e9e"

st.set_page_config(page_title="Signal History", layout="wide")
st.title("Signal History — SPY Signals & Contributions")

@st.cache_data(ttl=3600)
def fetch_spy(start: date, end: date) -> pd.DataFrame:
    import yfinance as yf
    # Normalize inputs to timestamps
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    df = pd.DataFrame()
    # Try multiple attempts: include end day by requesting end + 1, and small padding if needed
    for pad_days in (0, 1, 3):
        s = (start_ts - pd.Timedelta(days=pad_days)).strftime('%Y-%m-%d')
        # add 1 day to end so yf.download includes the requested end date
        e = (end_ts + pd.Timedelta(days=pad_days + 1)).strftime('%Y-%m-%d')
        try:
            tmp = yf.download("SPY", start=s, end=e, auto_adjust=False, progress=False)
            if tmp is None or tmp.empty:
                continue
            # if we have something, take it and break
            df = tmp.copy()
            break
        except Exception:
            continue

    # If download produced nothing, try using cached full-history from session_state if available
    if (df is None or df.empty) and 'spy_full' in st.session_state and st.session_state.get('spy_full') is not None:
        try:
            sf = st.session_state.get('spy_full')
            # If stored as list/dict (unexpected), convert to DataFrame
            if isinstance(sf, (list, tuple)):
                try:
                    sf = pd.DataFrame(sf)
                except Exception:
                    sf = None
            # proceed if we have a DataFrame-like
            if isinstance(sf, pd.DataFrame):
                sf = sf.copy()
                # build a Date column robustly
                if 'Date' in sf.columns:
                    sf['Date'] = pd.to_datetime(sf['Date'], errors='coerce')
                else:
                    # try index
                    try:
                        idx = pd.to_datetime(sf.index, errors='coerce')
                        if idx.notna().sum() / max(1, len(idx)) > 0.5:
                            sf = sf.reset_index()
                            if 'index' in sf.columns:
                                sf = sf.rename(columns={'index': 'Date'})
                                sf['Date'] = pd.to_datetime(sf['Date'], errors='coerce')
                    except Exception:
                        pass
                    # try to find any datetime-like column
                    if 'Date' not in sf.columns:
                        for c in sf.columns:
                            try:
                                coerced = pd.to_datetime(sf[c], errors='coerce')
                                if coerced.notna().sum() / max(1, len(coerced)) > 0.6:
                                    sf[c] = coerced
                                    sf = sf.rename(columns={c: 'Date'})
                                    break
                            except Exception:
                                continue
                # if we have Date, filter
                if 'Date' in sf.columns:
                    sf['Date'] = pd.to_datetime(sf['Date'])
                    mask_sf = (sf['Date'] >= start_ts) & (sf['Date'] <= end_ts)
                    sf_slice = sf.loc[mask_sf].reset_index(drop=True)
                    if not sf_slice.empty:
                        df = sf_slice.copy()
        except Exception:
            pass

    if df is None or df.empty:
        return pd.DataFrame()

    # Handle multi-level column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col]).strip('_') for col in df.columns]

    # Ensure Date column exists and is a column (not index), then filter to the exact requested window
    # Ensure DataFrame has a 'Date' column. Reset index and try to detect a datetime-like column.
    if 'Date' not in df.columns:
        try:
            df = df.reset_index()
        except Exception:
            pass
        # If still no 'Date', try to find a datetime column and rename it
        if 'Date' not in df.columns:
            # common fallback names
            if 'index' in df.columns and pd.api.types.is_datetime64_any_dtype(df['index']):
                df = df.rename(columns={'index': 'Date'})
            else:
                # look for any datetime-like column
                datetime_col = None
                for c in df.columns:
                    try:
                        if pd.api.types.is_datetime64_any_dtype(df[c]):
                            datetime_col = c
                            break
                        # try coercion without raising
                        coerced = pd.to_datetime(df[c], errors='coerce')
                        if coerced.notna().sum() > 0 and coerced.notna().sum() / len(coerced) > 0.6:
                            df[c] = coerced
                            datetime_col = c
                            break
                    except Exception:
                        continue
                if datetime_col:
                    df = df.rename(columns={datetime_col: 'Date'})
    # final guard
    if 'Date' not in df.columns:
        return pd.DataFrame()

    # Create a reliable datetime Series for filtering (prefer column, fallback to index)
    try:
        if 'Date' in df.columns:
            dates_series = pd.to_datetime(df['Date'], errors='coerce')
        else:
            dates_series = pd.to_datetime(df.index, errors='coerce')
    except Exception:
        # last-resort: coerce everything
        dates_series = pd.to_datetime(df.index, errors='coerce')

    # Assign normalized Date column and drop rows where Date failed to parse
    df['Date'] = dates_series
    df = df.dropna(subset=['Date']).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    # Now safe to sort and continue
    df = df.sort_values('Date').reset_index(drop=True)

    # Find price column
    candidates = [c for c in df.columns
                  if ("adj close" in str(c).lower()) or (str(c).lower() == "close") or str(c).lower().endswith("_close")]
    if not candidates:
        # As a fallback, try common numeric column names
        numcols = df.select_dtypes('number').columns
        if len(numcols):
            price_col = numcols[0]
        else:
            return pd.DataFrame()
    else:
        price_col = candidates[0]

    df.rename(columns={price_col: "Price"}, inplace=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").astype("float32")

    # filter to requested date range (inclusive)
    mask = (pd.to_datetime(df['Date']) >= start_ts) & (pd.to_datetime(df['Date']) <= end_ts)
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    df["ret"] = df["Price"].pct_change().astype("float32")
    df = df.dropna(subset=["Price"]).reset_index(drop=True)
    return df

# Default signal weights (same defaults used elsewhere)
DEFAULT_WEIGHTS = {
    'price_lt_ema50': 0.15,
    'ema20_lt_ema50': 0.10,
    'mom21_lt_0': 0.10,
    'atr_gt_sma63': 0.10,
    'rv20_gt_rv63': 0.10,
    'vix_term_pos': 0.15,
    'rsp_spy_63_neg': 0.15,
    'hyg_lqd_21_neg': 0.10,
    'hmm_bear_prob': 0.15
}

# Compute signals (vectorized) for a DataFrame with columns Date, Price, ret
def compute_signals(df: pd.DataFrame, extras: dict) -> pd.DataFrame:
    """Return DataFrame of signal columns aligned to df.index.
    extras: dict of ticker -> DataFrame or Series (indexed by Date) or empty frame.
    This function is defensive: it tolerates missing extras and different column names.
    """
    out = pd.DataFrame(index=df.index)

    # helper: safely extract a price Series from extras for a given key
    def _get_price_series(key):
        s = extras.get(key)
        if s is None or (isinstance(s, pd.DataFrame) and s.empty):
            return None
        # If it's a DataFrame, prefer 'Price' or first numeric column; ensure datetime index
        if isinstance(s, pd.DataFrame):
            # try column names
            if 'Price' in s.columns:
                ser = s['Price']
            else:
                numcols = s.select_dtypes('number').columns
                if len(numcols) > 0:
                    ser = s[numcols[0]]
                else:
                    # maybe it's already indexed with Price as a column name mismatch
                    # try to coerce any column to numeric and pick the one with fewest NaNs
                    best = None; best_non_na = -1
                    for c in s.columns:
                        try:
                            coer = pd.to_numeric(s[c], errors='coerce')
                            non_na = coer.notna().sum()
                            if non_na > best_non_na:
                                best_non_na = non_na; best = coer
                        except Exception:
                            continue
                    if best is None or best_non_na == 0:
                        return None
                    ser = best
            # ensure index is datetime
            try:
                ser = ser.copy()
                if not pd.api.types.is_datetime64_any_dtype(ser.index):
                    ser.index = pd.to_datetime(s.index, errors='coerce')
            except Exception:
                try:
                    ser.index = pd.to_datetime(ser.index.astype(str), errors='coerce')
                except Exception:
                    pass
            # drop NaT in index
            ser = ser[~ser.index.isna()]
            if ser.empty:
                return None
            return ser
        # If it's a Series
        if isinstance(s, pd.Series):
            ser = s.copy()
            if not pd.api.types.is_datetime64_any_dtype(ser.index):
                try:
                    ser.index = pd.to_datetime(ser.index, errors='coerce')
                except Exception:
                    pass
            ser = ser[~ser.index.isna()]
            if ser.empty:
                return None
            return ser
        return None

    # Basic required series on df
    if 'Price' not in df.columns:
        # attempt to find a numeric price-like column
        numcols = df.select_dtypes('number').columns
        # look for common price column names (case-insensitive)
        candidates = [c for c in df.columns if any(k in str(c).lower() for k in ['price','close','adj close','adj_close'])]
        if candidates:
            df = df.rename(columns={candidates[0]: 'Price'})
        elif len(numcols) and 'ret' in df.columns:
            # fallback to first numeric column
            df['Price'] = pd.to_numeric(df[numcols[0]], errors='coerce')
        else:
            # Provide user-friendly Streamlit error and stop the app run
            st.error("SPY price column not found in data. Ensure SPY data contains a Price/Close column or refresh full history.")
            st.stop()

    price = df['Price'].astype('float32')
    out['price_lt_ema50'] = (price < price.ewm(span=50, adjust=False).mean()).astype(float)
    out['ema20_lt_ema50'] = (price.ewm(span=20, adjust=False).mean() < price.ewm(span=50, adjust=False).mean()).astype(float)
    out['mom21_lt_0'] = (price.diff(21) < 0).astype(float)

    # ATR approx
    if {'High', 'Low', 'Close'}.issubset(df.columns):
        h = pd.to_numeric(df['High'], errors='coerce')
        l = pd.to_numeric(df['Low'], errors='coerce')
        c = pd.to_numeric(df['Close'], errors='coerce')
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
    else:
        tr = price.rolling(2).apply(lambda x: np.nan if getattr(x, 'size', len(x)) < 2 else float(abs(x[-1] - x[0])), raw=True)
        atr14 = tr.rolling(14).mean()
    atr63 = atr14.rolling(63).mean()
    out['atr_gt_sma63'] = ((atr14 / price) > (atr63 / price.replace(0, np.nan))).astype(float)

    # realized vol
    rv20 = df['ret'].rolling(20).std()
    rv63 = df['ret'].rolling(63).std()
    out['rv20_gt_rv63'] = (rv20 > rv63).astype(float)

    # term structure: VIX - VIX3M
    vix_ser = _get_price_series('^VIX')
    vix3m_ser = _get_price_series('^VIX3M')
    if vix_ser is not None and vix3m_ser is not None:
        # align to df dates
        idx = pd.to_datetime(df['Date'])
        term = vix_ser.reindex(idx).ffill(limit=1) - vix3m_ser.reindex(idx).ffill(limit=1)
        out['vix_term_pos'] = (term > 0).astype(float)
    else:
        out['vix_term_pos'] = np.nan

    # breadth: RSP/SPY 63-day return < 0
    rsp_ser = _get_price_series('RSP')
    if rsp_ser is not None:
        idx = pd.to_datetime(df['Date'])
        rsp_aligned = rsp_ser.reindex(idx).astype('float32')
        ratio_return63 = (rsp_aligned / df['Price']).pct_change(63)
        out['rsp_spy_63_neg'] = (ratio_return63 < 0).astype(float)
    else:
        out['rsp_spy_63_neg'] = np.nan

    # credit proxy: (HYG/LQD) 21-day return < 0
    hyg_ser = _get_price_series('HYG')
    lqd_ser = _get_price_series('LQD')
    if hyg_ser is not None and lqd_ser is not None:
        idx = pd.to_datetime(df['Date'])
        hyg_a = hyg_ser.reindex(idx).astype('float32')
        lqd_a = lqd_ser.reindex(idx).astype('float32')
        credit_ratio21 = (hyg_a / lqd_a).pct_change(21)
        out['hyg_lqd_21_neg'] = (credit_ratio21 < 0).astype(float)
    else:
        out['hyg_lqd_21_neg'] = np.nan

    # HMM Bear probability from session state (already handled elsewhere; defensively map)
    hb_raw = st.session_state.get('hmm_bear_prob_series')
    if hb_raw:
        try:
            hb_df = pd.DataFrame(hb_raw)
            hb_df['Date'] = pd.to_datetime(hb_df['Date']).dt.normalize()
            hb_df = hb_df.set_index('Date')
            hb_series = hb_df['Value'].reindex(pd.to_datetime(df['Date'])).ffill().fillna(0.5)
            out['hmm_bear_prob'] = hb_series.values.astype(float)
        except Exception:
            out['hmm_bear_prob'] = 0.5
    else:
        out['hmm_bear_prob'] = 0.5

    # ensure float32 where safe
    return out.astype('float32')

# Helper: compute weighted score per row using DEFAULT_WEIGHTS, normalize per available components
def compute_weighted_scores(signals_df: pd.DataFrame, weights_map: dict) -> (pd.Series, pd.DataFrame):
    comp_names = signals_df.columns.tolist()
    base_w = np.array([weights_map.get(c, 0.0) for c in comp_names], dtype=float)
    vals = signals_df.values.astype(float)
    # mask NA
    valid = ~np.isnan(vals)
    W = np.zeros_like(vals, dtype=float)
    for i in range(len(vals)):
        active = valid[i].astype(float)
        w = base_w * active
        s = w.sum()
        if s <= 0:
            if active.sum() > 0:
                w = active / active.sum()
            else:
                w = np.zeros_like(w)
        else:
            w = w / s
        W[i, :] = w
    contrib = np.nan_to_num(vals, nan=0.0) * W
    score01 = contrib.sum(axis=1)
    score100 = (score01 * 100.0).astype(float)
    contrib_df = pd.DataFrame(contrib, columns=comp_names, index=signals_df.index)
    return pd.Series(score100, index=signals_df.index), contrib_df

# UI inputs
st.sidebar.header('Signal History — Controls')
# date range default: last 5 years
today = date.today()
default_start = date(today.year-5, today.month, today.day)
start = st.sidebar.date_input('Start', value=default_start)
end = st.sidebar.date_input('End', value=today)
page_size = st.sidebar.selectbox('Rows per page', [25, 50, 100, 250], index=1)

# fetch data
with st.spinner('Loading SPY and extras...'):
    spy = fetch_spy(start, end)
    # extras: VIX, VIX3M, RSP, HYG, LQD
    extras = {}
    for t in ['^VIX','^VIX3M','RSP','HYG','LQD']:
        df_t = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = ['_'.join([str(c) for c in col]).strip('_') for col in df_t.columns]
        if df_t is None or df_t.empty:
            extras[t if t.startswith('^') else t] = pd.DataFrame()
        else:
            df_t = df_t.reset_index().sort_values('Date').reset_index(drop=True)
            price_col = None
            for cand in ['Adj Close','Adj_Close','Close']:
                if cand in df_t.columns:
                    price_col = cand; break
            if price_col is None:
                extras[t if t.startswith('^') else t] = pd.DataFrame()
            else:
                df_t['Price'] = pd.to_numeric(df_t[price_col], errors='coerce').astype('float32')
                key = t if t.startswith('^') else t
                # create a Series with datetime index for simpler alignment
                ser = pd.Series(df_t['Price'].values, index=pd.to_datetime(df_t['Date']), name='Price')
                # drop NaNs
                ser = ser[~ser.index.isna()].astype('float32')
                extras[key] = ser

if spy.empty:
    st.error('No SPY data for selected window.')
    st.stop()

# compute signals
signals = compute_signals(spy, extras)
# compute weighted score (drop contributions — UI does not need them)
score_series, _ = compute_weighted_scores(signals, DEFAULT_WEIGHTS)

# Combine into a table (signals + total score only)
table = pd.DataFrame({'Date': pd.to_datetime(spy['Date']).dt.date})
for col in signals.columns:
    table[col] = signals[col].replace({0.0:0,1.0:1}).values
# total score
table['Score(0-100)'] = score_series.round(2).values

# Sort newest first and paginate
table = table.sort_values('Date', ascending=False).reset_index(drop=True)
n_rows = len(table)
n_pages = max(1, int(np.ceil(n_rows / page_size)))
page = st.sidebar.number_input('Page', min_value=1, max_value=n_pages, value=1, step=1)
start_idx = (page-1)*page_size
end_idx = min(n_rows, page*page_size)

st.markdown(f"Showing rows {start_idx+1}–{end_idx} of {n_rows}")

# prepare display slice (newest first)
display = table.iloc[start_idx:end_idx].copy()
# format signals as check marks
for col in signals.columns:
    display[col] = display[col].map({1: '✅', 0: '❌', np.nan: 'NA'})

# color scale for score bands
def score_color(val):
    try:
        v = float(val)
    except Exception:
        return ''
    if v < 40:
        return 'background-color: #1e7f54; color: white'
    if v < 60:
        return 'background-color: #bdb76b; color: black'
    if v < 80:
        return 'background-color: #ff8c00; color: black'
    return 'background-color: #b33a3a; color: white'

# Render table using st.dataframe with pandas Styler
styler = display.style
# highlight score column
if 'Score(0-100)' in display.columns:
    styler = styler.applymap(lambda v: score_color(v) if isinstance(v,(int,float)) else '', subset=['Score(0-100)'])
# make numeric columns right-aligned via format
numeric_cols = [c for c in display.columns if display[c].dtype in [np.float64, np.int64]]
for c in numeric_cols:
    styler = styler.format({c: "{:.4f}"})

# compute dataframe height so the selected page_size shows without excessive scrolling
row_height = 26  # approx pixels per row
header_height = 110
max_height = 1200
height = min(max_height, header_height + row_height * page_size)
st.dataframe(styler, use_container_width=True, height=int(height))

# CSV download for the full (filtered) range
csv_buf = io.StringIO()
# save full table for download (not just page) but respecting memory: use to_csv on filtered dates
filtered_start = st.sidebar.date_input('Download: start', value=start)
filtered_end = st.sidebar.date_input('Download: end', value=end)
mask = (pd.to_datetime(table['Date']) >= pd.to_datetime(filtered_start)) & (pd.to_datetime(table['Date']) <= pd.to_datetime(filtered_end))
dl = table.loc[mask]
if st.sidebar.button('Download CSV'):
    csv_buf.write(dl.to_csv(index=False))
    st.sidebar.download_button('Download CSV', data=csv_buf.getvalue(), file_name='signal_history.csv', mime='text/csv')

st.sidebar.markdown('---')
st.sidebar.caption('Signal History is paginated for memory efficiency.')

# end of page
