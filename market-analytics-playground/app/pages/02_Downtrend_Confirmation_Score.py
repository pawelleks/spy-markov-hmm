# New Streamlit page: Downtrend Confirmation Score
# This page is additive and does not modify other app pages.
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime
import matplotlib.dates as mdates
import plotly.graph_objects as go

# --- Theme/colors & helpers (minimal, copied to avoid importing app.py which may execute) ---
DARK_BG = "#0b1220"
DARK_AX = "#0e1525"
GRID = "#203049"
FG = "#d7e3f3"
LINE = "#9ec4ff"
GREEN = "#4caf50"
RED = "#f44336"
NEUTRAL = "#9e9e9e"

def styled_fig(size=(6, 3)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID, alpha=0.6, linewidth=0.6)
    return fig, ax

def downsample_ts(dates: pd.Series, values: pd.Series, max_pts: int = 2500):
    n = len(values)
    if n <= max_pts or n == 0:
        return dates, values
    step = int(np.ceil(n / max_pts))
    return dates.iloc[::step].reset_index(drop=True), values.iloc[::step].reset_index(drop=True)

# Helper: downsample (preserve alignment) for plotting large series
def downsample_for_plot(dates: pd.DatetimeIndex, values: pd.Series, max_pts: int = 2500):
    n = len(values)
    if n <= max_pts or n == 0:
        return dates, values
    step = int(np.ceil(n / max_pts))
    return dates[::step], values.iloc[::step]

# Robust field getter for mixed dict key styles (Score vs score vs Score (0-1))
def get_field(it, keys, default=np.nan):
    if isinstance(it, dict):
        for k in keys:
            if k in it:
                return it[k]
    return default

# Small helper to produce status badge text and color for a component score (0..1 or 0..100)
def status_badge(v):
    if v is None:
        return ('NA', NEUTRAL)
    try:
        vv = float(v)
    except Exception:
        return ('NA', NEUTRAL)
    # normalize if on 0..100 scale
    if vv > 1.0:
        vv = vv / 100.0
    if np.isnan(vv):
        return ('NA', NEUTRAL)
    if vv <= 0.33:
        return ('OK', '#1e7f54')
    if vv <= 0.66:
        return ('CAUTION', '#b08b00')
    return ('RISK', '#b33a3a')

# Build Plotly figure for downtrend score
def build_downtrend_plotly(dates, score_raw, score_smooth, warn, alert, crisis, show_raw=True):
    # Colors and style
    bg_panel = DARK_BG
    bg_plot = DARK_AX
    fg = FG
    raw_color = 'rgba(160,195,255,0.25)'
    sm_color = 'rgb(158,196,255)'
    warn_col = 'rgba(255,215,0,0.6)'
    alert_col = 'rgba(255,99,71,0.7)'
    crisis_col = 'rgba(220,20,60,0.8)'

    # Downsample raw for performance
    ds_dates_raw, ds_raw = downsample_for_plot(dates, score_raw, max_pts=2500)

    fig = go.Figure()

    # Raw series (thin faint line / markers)
    if show_raw:
        fig.add_trace(go.Scatter(
            x=ds_dates_raw,
            y=ds_raw,
            mode='lines',
            name='Raw',
            line=dict(color=raw_color, width=0.6),
            hoverinfo='skip',
            opacity=0.25,
            showlegend=True
        ))

    # Smoothed: glow then main line
    # glow
    fig.add_trace(go.Scatter(
        x=dates,
        y=score_smooth,
        mode='lines',
        name='_glow',
        line=dict(color=sm_color, width=6),
        opacity=0.18,
        hoverinfo='skip',
        showlegend=False
    ))
    # main
    fig.add_trace(go.Scatter(
        x=dates,
        y=score_smooth,
        mode='lines',
        name=f'Smoothed',
        line=dict(color=sm_color, width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>%{y:.1f}<extra></extra>',
        showlegend=True
    ))

    # Optional 20-day rolling min/max band (very subtle)
    try:
        roll_max = score_raw.rolling(20, min_periods=1).max()
        roll_min = score_raw.rolling(20, min_periods=1).min()
        fig.add_trace(go.Scatter(
            x=pd.concat([dates, dates[::-1]]),
            y=pd.concat([roll_max, roll_min[::-1]]),
            fill='toself',
            fillcolor='rgba(158,196,255,0.02)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            showlegend=False
        ))
    except Exception:
        pass

    # Threshold horizontal lines as shapes and annotations
    shapes = []
    annotations = []
    # helper for annotation x position: right edge
    xref = 'paper'
    xanchor = 1.01
    for th, label, color in [(warn, 'Warning', warn_col), (alert, 'Alert', alert_col), (crisis, 'Crisis', crisis_col)]:
        shapes.append(dict(type='line', xref='paper', x0=0, x1=1, y0=th, y1=th, line=dict(color=color, width=1, dash='dash')))
        annotations.append(dict(xref=xref, x=xanchor, y=th, xanchor='left', yanchor='middle', text=f'{label} {int(th)}', showarrow=False, font=dict(color=color, size=10)))

    fig.update_layout(
        template=None,
        paper_bgcolor=bg_panel,
        plot_bgcolor=bg_plot,
        margin=dict(l=50, r=120, t=60, b=40),
        shapes=shapes,
        annotations=annotations,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.02, xanchor='left', x=0.01, font=dict(color=fg)),
        title=dict(text='Downtrend Confirmation Score (Historical)', font=dict(color=fg, size=14), x=0.01)
    )

    # Axes
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=0.5, zeroline=False, color=fg)
    fig.update_yaxes(range=[0, 105], dtick=20, showgrid=True, gridcolor=GRID, gridwidth=0.5, zeroline=False, color=fg, ticks='outside')

    # Range selector + slider
    fig.update_layout(
        xaxis=dict(rangeselector=dict(buttons=list([
            dict(count=1, label='1m', step='month', stepmode='backward'),
            dict(count=3, label='3m', step='month', stepmode='backward'),
            dict(count=6, label='6m', step='month', stepmode='backward'),
            dict(count=1, label='YTD', step='year', stepmode='todate'),
            dict(count=1, label='1y', step='year', stepmode='backward'),
            dict(count=5, label='5y', step='year', stepmode='backward'),
            dict(step='all', label='All')
        ])), rangeslider=dict(visible=True), type='date')
    )

    # interactive and visual tweaks
    fig.update_layout(dragmode='zoom')

    return fig

# --- Data download helpers ---
@st.cache_data(show_spinner=False, ttl="6h")
def fetch_symbol(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download a single ticker using yfinance and return a clean DataFrame with Date, Open, High, Low, Close, Adj Close, Volume, Price, ret."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col]).strip("_") for col in df.columns]
    df = df.reset_index().sort_values("Date").reset_index(drop=True)
    # prefer Adj Close
    price_col = None
    for cand in ["Adj Close", "Adj_Close", "Close"]:
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None and "Close" in df.columns:
        price_col = "Close"
    if price_col:
        df["Price"] = pd.to_numeric(df[price_col], errors="coerce").astype("float32")
    else:
        # fallback: take first numeric column
        numcols = df.select_dtypes("number").columns
        if len(numcols):
            df["Price"] = pd.to_numeric(df[numcols[0]], errors="coerce").astype("float32")
        else:
            return pd.DataFrame()
    # ensure High/Low/Close exist for ATR
    for c in ["High", "Low", "Close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["ret"] = df["Price"].pct_change().astype("float32")
    return df

@st.cache_data(show_spinner=False, ttl="6h")
def fetch_multiple(tickers: list, start: date, end: date) -> dict:
    """Fetch multiple symbols individually and return dict of DataFrames keyed by ticker."""
    out = {}
    for t in tickers:
        out[t] = fetch_symbol(t, start, end)
    return out

# --- Page UI ---
st.set_page_config(page_title="Downtrend Confirmation Score", layout="wide")
st.markdown("# Downtrend Confirmation Score")

# Use same default window as main app
today = date.today()
if "start_date" in st.session_state and "end_date" in st.session_state:
    sd = st.session_state.get("start_date")
    ed = st.session_state.get("end_date")
    # session_state may store python dates or strings
    try:
        if isinstance(sd, str):
            sd = datetime.fromisoformat(sd).date()
        if isinstance(ed, str):
            ed = datetime.fromisoformat(ed).date()
    except Exception:
        sd = date(1993, 1, 1)
        ed = today
else:
    sd = date(1993, 1, 1)
    ed = today

with st.sidebar:
    st.header("Downtrend Score — Data & Settings")
    st.date_input("Start date", value=sd, key="dcs_start")
    st.date_input("End date", value=ed, key="dcs_end")

start = st.session_state.get("dcs_start", sd)
end = st.session_state.get("dcs_end", ed)

# fetch SPY and extras
tickers = ["SPY", "^VIX", "^VIX3M", "RSP", "HYG", "LQD"]
with st.spinner("Fetching price data (may take a few seconds)..."):
    data_dict = fetch_multiple(tickers, start, end)

spy = data_dict.get("SPY", pd.DataFrame())
if spy.empty:
    st.error("Could not download SPY data for the selected window.")
    st.stop()

# Merge data on Date
# Start with SPY
df = spy[["Date", "Price", "ret"]].rename(columns={"Price": "SPY_Price", "ret": "SPY_ret"}).copy()
# for each extra ticker, merge Price as TICK_Price, keep High/Low/Close where present
for t in ["^VIX", "^VIX3M", "RSP", "HYG", "LQD"]:
    d = data_dict.get(t, pd.DataFrame())
    col_price = f"{t}_Price"
    if d is None or d.empty:
        st.info(f"Ticker {t} not available; its component will be excluded.")
        df[col_price] = np.nan
        continue
    tmp = d[["Date", "Price"]].rename(columns={"Price": col_price})
    df = df.merge(tmp, on="Date", how="outer")

# sort and forward-fill only for VIX and VIX3M single-day missing
df = df.sort_values("Date").reset_index(drop=True)
# forward-fill limit=1 for vix series
for v in ["^VIX_Price", "^VIX3M_Price"]:
    if v in df.columns:
        df[v] = df[v].ffill(limit=1)
# other tickers: keep as-is (may have NA)

# ensure SPY_ret exists
if "SPY_ret" not in df.columns:
    df["SPY_ret"] = df["SPY_Price"].pct_change().astype("float32")

# compute EMAs and momentum on SPY
spy_price = df["SPY_Price"].astype("float32")
ema50 = spy_price.ewm(span=50, adjust=False).mean()
ema20 = spy_price.ewm(span=20, adjust=False).mean()
mom21 = spy_price.diff(21)

# ATR(14)
# Need SPY High/Low/Close — fetch from SPY data dict if available
spy_full = data_dict.get("SPY")
if spy_full is not None and not spy_full.empty and all(c in spy_full.columns for c in ["High", "Low", "Close"]):
    h = spy_full["High"].astype("float32").reset_index(drop=True)
    l = spy_full["Low"].astype("float32").reset_index(drop=True)
    c = spy_full["Close"].astype("float32").reset_index(drop=True)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
else:
    # approximate ATR with rolling high-low if missing
    if "SPY_Price" in df.columns:
        # approximate using rolling range
        # Use raw=True to get numpy arrays (avoid Series label-based indexing which can raise KeyError for -1)
        tr = spy_price.rolling(2).apply(lambda arr: np.nan if getattr(arr, "size", len(arr)) < 2 else float(abs(arr[-1] - arr[0])), raw=True)
        atr14 = tr.rolling(14).mean()
    else:
        atr14 = pd.Series(np.nan, index=df.index)

# realized vol
rv20 = df["SPY_ret"].rolling(20).std()
rv63 = df["SPY_ret"].rolling(63).std()
rv20 = rv20.astype("float32")
rv63 = rv63.astype("float32")

# term structure: VIX - VIX3M
vix = df.get("^VIX_Price")
vix3m = df.get("^VIX3M_Price")
term_diff = None
if vix is not None and vix3m is not None:
    term_diff = (vix - vix3m).astype("float32")

# breadth proxy: RSP/SPY 63-day return < 0
if "RSP_Price" in df.columns:
    rsp = df["RSP_Price"].astype("float32")
    rsp63 = rsp.pct_change(63)
    spy63 = df["SPY_Price"].astype("float32").pct_change(63)
    breadth_pct63 = (rsp / df["SPY_Price"]) - 1.0
    # but requirement: (RSP/SPY) 63-day return < 0 -> compute returns of ratio
    ratio_return63 = (rsp / df["SPY_Price"]).pct_change(63)
else:
    ratio_return63 = None

# credit proxy: (HYG/LQD) 21-day return < 0
if "HYG_Price" in df.columns and "LQD_Price" in df.columns:
    hyg = df["HYG_Price"].astype("float32")
    lqd = df["LQD_Price"].astype("float32")
    credit_ratio21 = (hyg / lqd).pct_change(21)
else:
    credit_ratio21 = None

# HMM Bear probability from session state
hmm_bear = None
hb_raw = st.session_state.get("hmm_bear_prob_series")
if hb_raw is not None:
    # 1. Try to interpret as list of dicts (records) -> DataFrame
    #    Expected format: [{'Date': timestamp, 'Value': float}, ...]
    try:
        if isinstance(hb_raw, list) and len(hb_raw) > 0 and isinstance(hb_raw[0], dict):
            hb_df = pd.DataFrame(hb_raw)
            if {"Date", "Value"}.issubset(hb_df.columns):
                hb_df["Date"] = pd.to_datetime(hb_df["Date"]).dt.normalize()
                # Reindex to match the current page's dataframe dates
                # Use ffill() to propagate the last known HMM state forward, and bfill() for initial gap
                tmp = hb_df.set_index("Date")["Value"].reindex(df["Date"]).ffill().bfill()
                hmm_bear = tmp.astype("float32")
    except Exception:
        pass

    # 2. If that failed or wasn't a dict list, try as simple array aligned by position
    if hmm_bear is None:
        try:
            hb_arr = np.asarray(hb_raw, dtype=float)
            if len(hb_arr) == len(df):
                hmm_bear = pd.Series(hb_arr, index=df.index).astype("float32")
        except Exception:
            hmm_bear = None

if hmm_bear is None:
    st.warning("HMM Bear Probability model not found in session state. Please visit the **Home** page to initialize the model.")


# --- Signals (0/1 or NA) ---
signals = pd.DataFrame(index=df.index)
signals["price_lt_ema50"] = (spy_price < ema50).astype(float)
signals["ema20_lt_ema50"] = (ema20 < ema50).astype(float)
signals["mom21_lt_0"] = (mom21 < 0).astype(float)
# ATR relative to its 63-day sma
atr63 = pd.Series(atr14).rolling(63).mean()
signals["atr_gt_sma63"] = (atr14 / spy_price > atr63 / spy_price.replace(0, np.nan)).astype(float)
signals["rv20_gt_rv63"] = (rv20 > rv63).astype(float)
if term_diff is not None:
    signals["vix_term_pos"] = (term_diff > 0).astype(float)
else:
    signals["vix_term_pos"] = np.nan
if ratio_return63 is not None:
    signals["rsp_spy_63_neg"] = (ratio_return63 < 0).astype(float)
else:
    signals["rsp_spy_63_neg"] = np.nan
if credit_ratio21 is not None:
    signals["hyg_lqd_21_neg"] = (credit_ratio21 < 0).astype(float)
else:
    signals["hyg_lqd_21_neg"] = np.nan
if hmm_bear is not None:
    # use raw probability (0..1) as the component
    signals["hmm_bear_prob"] = hmm_bear
else:
    signals["hmm_bear_prob"] = np.nan

# Cast to float32
signals = signals.astype("float32")

# Sidebar: weights
with st.sidebar.expander("Downtrend Score Settings", expanded=True):
    st.markdown("**Component weights (editable)**")
    w_price_lt_ema50 = st.slider("Price < EMA50", 0.0, 1.0, 0.15, step=0.01)
    w_ema20_lt_ema50 = st.slider("EMA20 < EMA50", 0.0, 1.0, 0.10, step=0.01)
    w_mom21 = st.slider("Momentum21 < 0", 0.0, 1.0, 0.10, step=0.01)
    w_atr = st.slider("ATR rising (ATR/Price > sma63)", 0.0, 1.0, 0.10, step=0.01)
    w_rv = st.slider("Realized Vol rising", 0.0, 1.0, 0.10, step=0.01)
    w_vix = st.slider("VIX - VIX3M > 0", 0.0, 1.0, 0.15, step=0.01)
    w_rsp = st.slider("RSP/SPY 63d < 0", 0.0, 1.0, 0.15, step=0.01)
    w_hyg = st.slider("HYG/LQD 21d < 0", 0.0, 1.0, 0.10, step=0.01)
    w_hmm = st.slider("HMM Bear prob (if available)", 0.0, 1.0, 0.15, step=0.01)

weights_df = pd.Series({
    "price_lt_ema50": w_price_lt_ema50,
    "ema20_lt_ema50": w_ema20_lt_ema50,
    "mom21_lt_0": w_mom21,
    "atr_gt_sma63": w_atr,
    "rv20_gt_rv63": w_rv,
    "vix_term_pos": w_vix,
    "rsp_spy_63_neg": w_rsp,
    "hyg_lqd_21_neg": w_hyg,
    "hmm_bear_prob": w_hmm,
}).astype("float32")

# Thresholds
with st.sidebar.expander("Thresholds", expanded=False):
    warn_th = st.slider("Warning threshold", 0, 100, 40, step=1)
    alert_th = st.slider("Alert threshold", 0, 100, 60, step=1)
    crisis_th = st.slider("Crisis threshold", 0, 100, 80, step=1)

# --- Scoring: per-date normalization of weights excluding NA components ---
comp_names = signals.columns.tolist()
W = np.zeros((len(df), len(comp_names)), dtype=np.float32)
base_weights = weights_df.reindex(comp_names).fillna(0).values.astype(np.float32)
# Broadcast base weights then zero-out where signals are NA
sig_vals = signals.values.astype(np.float32)
valid_mask = ~np.isnan(sig_vals)
# For each date, compute active weight sum and normalize
for i in range(len(df)):
    active = valid_mask[i]
    if active.sum() == 0:
        W[i, :] = 0.0
    else:
        w = base_weights * active.astype(np.float32)
        s = w.sum()
        if s <= 0:
            # distribute equally among active
            w = active.astype(np.float32) / active.sum()
        else:
            w = w / s
        W[i, :] = w

# Weighted score (0..1)
weighted = (np.nan_to_num(sig_vals, nan=0.0) * W).sum(axis=1)
score_01 = weighted.astype(np.float32)
score_100 = (score_01 * 100.0).astype(np.float32)

# Create a pandas Series for convenience (supports .shift and boolean indexing)
score_s = pd.Series(score_100, index=df.index, name="downtrend_score")

# KPI for most recent available date
last_idx = np.where(~np.isnan(score_100))[0]
if len(last_idx):
    last = last_idx[-1]
    current_score = float(score_100[last])
else:
    current_score = float('nan')

# Status label
def status_label(val):
    if np.isnan(val):
        return "N/A", NEUTRAL
    if val >= crisis_th:
        return "CRISIS", RED
    if val >= alert_th:
        return "ALERT", RED
    if val >= warn_th:
        return "WARNING", "#ffb74d"
    return "OK", GREEN

status_text, status_color = status_label(current_score)

# Layout: KPI row
c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    st.metric("Downtrend Score (0–100)", f"{current_score:.1f}" if not np.isnan(current_score) else "N/A", delta=None)
with c2:
    st.markdown(f"### Status: <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
with c3:
    st.caption(f"Window: {start.isoformat()} → {end.isoformat()}")

# Component table for latest date (clean, robust implementation)
comp_table = []
human_names = {
    "price_lt_ema50": "Price < EMA(50)",
    "ema20_lt_ema50": "EMA(20) < EMA(50)",
    "mom21_lt_0": "21-day Momentum < 0",
    "atr_gt_sma63": "ATR(14)/Price > 63-day SMA",
    "rv20_gt_rv63": "Realized Vol (20d) > (63d)",
    "vix_term_pos": "VIX − VIX3M > 0 (term)",
    "rsp_spy_63_neg": "RSP/SPY 63d return < 0 (breadth)",
    "hyg_lqd_21_neg": "HYG/LQD 21d return < 0 (credit)",
    "hmm_bear_prob": "HMM: Bear probability"
}

# Build comp_table from current latest index (if available)
if len(last_idx):
    i = last
    total_weight = 0.0
    total_contrib = 0.0
    for j, name in enumerate(comp_names):
        rawv = sig_vals[i, j]
        score_val = np.nan if np.isnan(rawv) else float(rawv)
        weight_val = float(W[i, j])
        contrib_val = np.nan if np.isnan(rawv) else (float(rawv) * weight_val)
        total_weight += weight_val
        total_contrib += 0.0 if np.isnan(contrib_val) else contrib_val
        comp_table.append({
            'internal': name,
            'Signal': human_names.get(name, name),
            'Category': {
                'price_lt_ema50':'Trend','ema20_lt_ema50':'Trend','mom21_lt_0':'Trend',
                'atr_gt_sma63':'Volatility','rv20_gt_rv63':'Volatility',
                'vix_term_pos':'Market Structure','rsp_spy_63_neg':'Breadth',
                'hyg_lqd_21_neg':'Credit','hmm_bear_prob':'State Models'
            }.get(name,'Misc'),
            'Value/raw': ('NA' if np.isnan(rawv) else f"{score_val:.2f}"),
            'Score': (np.nan if np.isnan(score_val) else score_val),
            'Weight': weight_val,
            'Weighted contribution': (np.nan if np.isnan(contrib_val) else contrib_val)
        })
    # totals row
    comp_table.append({'internal':'sum','Signal':'SUM','Category':'','Value/raw':'','Score':'','Weight':total_weight,'Weighted contribution':total_contrib})

else:
    st.info("No valid scores available for the selected window.")

# Create a DataFrame copy for optional table rendering
df_comp_table = pd.DataFrame(comp_table)

# UI controls (single definition of each widget key)
show_explanations = st.sidebar.checkbox("Show explanations", value=True, key="dcs_show_explanations")
show_sparklines = st.sidebar.checkbox("Show sparklines", value=False, key="dcs_show_sparklines")
group_by_category = st.sidebar.checkbox("Group by category", value=True, key="dcs_group_by_category")
compact_mode = st.sidebar.checkbox("Compact mode (recommended)", value=True, key="dcs_compact_mode")
view_mode = st.sidebar.radio("View", ["Grid", "Table (dense)"], index=0, key="dcs_view_mode")

# small helpers for rendering
def make_comp_rows(comp_table):
    rows = []
    for it in comp_table:
        rows.append({
            'internal': it.get('internal',''),
            'signal': it.get('Signal',''),
            'category': it.get('Category',''),
            'score': (np.nan if it.get('Score')=='' else (np.nan if it.get('Score') is None else float(it.get('Score')))),
            'weight': float(it.get('Weight') or 0.0),
            'contrib': (np.nan if it.get('Weighted contribution')=='' else (np.nan if it.get('Weighted contribution') is None else float(it.get('Weighted contribution')))),
            'raw': it.get('Value/raw','')
        })
    return rows

rows = make_comp_rows(comp_table[:-1]) if len(comp_table)>0 else []

# Model confidence summary
available = [r for r in rows if not np.isnan(r['score'])]
active_count = sum(1 for r in available if r['score'] >= 0.5)
total_available = len(available)
if active_count <= 2:
    conf_txt, conf_col = ('Low', GREEN)
elif active_count <= 5:
    conf_txt, conf_col = ('Medium', '#b08b00')
else:
    conf_txt, conf_col = ('High', RED)

# Top summary strip
s1, s2, s3 = st.columns([1,1,1])
s1.markdown(f"**Confidence:** <span style='color:{conf_col}'>{conf_txt}</span>", unsafe_allow_html=True)
s2.markdown(f"**Active bearish signals:** <span style='color:{FG}'>{active_count} / {total_available}</span>", unsafe_allow_html=True)
score_display = f"{current_score:.1f}" if not np.isnan(current_score) else "N/A"
s3.markdown(f"**Score:** <span style='color:{FG}'>{score_display}</span>", unsafe_allow_html=True)

# Render Table (dense) or Grid
if view_mode == 'Table (dense)':
    if rows:
        df_table = pd.DataFrame(rows)
        df_table_display = df_table[['signal','score','weight','contrib']].rename(columns={'signal':'Signal','score':'Score','weight':'Weight','contrib':'Contribution'})
        df_table_display['Score'] = df_table_display['Score'].map(lambda x: 'NA' if np.isnan(x) else f"{x:.2f}")
        df_table_display['Weight'] = df_table_display['Weight'].map(lambda x: f"{x:.2f}")
        df_table_display['Contribution'] = df_table_display['Contribution'].map(lambda x: 'NA' if np.isnan(x) else f"{x:.2f}")
        st.dataframe(df_table_display.style.hide_index(), use_container_width=True, height=min(480, 28+24*len(df_table_display)))
    else:
        st.info("No component rows to display.")
else:
    # Grid mode (compact cards)
    if not rows:
        st.info("No component rows to display.")
    else:
        grouped = {}
        for r in rows:
            grouped.setdefault(r['category'], []).append(r)
        for cat, items in grouped.items():
            st.markdown(f"**{cat} — {sum(1 for it in items if not np.isnan(it['score']) and it['score']>=0.5)}/{len(items)} bearish**")
            ncols = 4
            chunks = [items[i:i+ncols] for i in range(0,len(items),ncols)]
            for chunk in chunks:
                cols = st.columns(ncols, gap='small')
                for col, it in zip(cols, chunk):
                    with col:
                        st.markdown(f"**{it['signal']}**")
                        st.caption(it['raw'])
                        pct = 'NA' if np.isnan(it['score']) else f"{it['score']:.2f}"
                        st.markdown(f"<div style='text-align:right;color:{FG}'>Score: {pct} • w={it['weight']:.2f}</div>", unsafe_allow_html=True)
                        stat, colr = status_badge(it['score'] if not np.isnan(it['score']) else np.nan)
                        st.markdown(f"<div style='background:{colr};color:{DARK_BG};padding:4px;border-radius:4px;text-align:center'>{stat}</div>", unsafe_allow_html=True)

# Totals
try:
    st.markdown('---')
    st.markdown(f"**Totals — Weight:** {sum([r['weight'] for r in rows]):.2f} — **Weighted contribution:** {sum([0.0 if np.isnan(r['contrib']) else r['contrib'] for r in rows]):.2f}")
except Exception:
    pass

# --- Charts ---
st.markdown("---")
st.markdown("### Downtrend Confirmation Score (Historical)")

# Chart controls: zoom window only in sidebar (keep smoothing selectbox above chart)
with st.sidebar.expander("Chart options", expanded=False):
    # date-range zoom: default to last 5 years for better default focus
    try:
        all_dates = pd.to_datetime(df["Date"]).dt.normalize()
        default_end = all_dates.max().date()
        default_start = (all_dates.max() - pd.DateOffset(years=5)).date()
    except Exception:
        default_start = pd.to_datetime(df["Date"]).min().date()
        default_end = pd.to_datetime(df["Date"]).max().date()
    zoom_start = st.date_input("Chart: start", value=default_start, key="dcs_zoom_start")
    zoom_end = st.date_input("Chart: end", value=default_end, key="dcs_zoom_end")

# Use datetime index and a pandas Series for the score so downsampling preserves alignment
dates = pd.to_datetime(df["Date"]).dt.normalize()
score_series = pd.Series(score_100, index=dates)

# Provide smoothing selectbox above chart and raw toggle (single authoritative widget)
smooth_choice_main = st.selectbox('Smoothing period (days)', options=[5, 10, 20], index=0, key='dcs_smooth_main')
show_raw = st.checkbox('Show raw series', value=True)

# Prepare data for plotting (use zoom mask as before)
try:
    z0 = pd.to_datetime(st.session_state.get('dcs_zoom_start', zoom_start))
    z1 = pd.to_datetime(st.session_state.get('dcs_zoom_end', zoom_end))
    zoom_mask = (dates >= z0) & (dates <= z1)
    plot_dates = dates[zoom_mask]
    plot_raw = score_series[zoom_mask]
    plot_smooth = plot_raw.rolling(window=int(smooth_choice_main), min_periods=1).mean()
except Exception:
    plot_dates = dates
    plot_raw = score_series
    plot_smooth = plot_raw.rolling(window=int(smooth_choice_main), min_periods=1).mean()

# Build and render plotly figure
fig_plotly = build_downtrend_plotly(plot_dates, plot_raw, plot_smooth, warn_th, alert_th, crisis_th, show_raw=show_raw)
plotly_config = {"displayModeBar": True, "scrollZoom": True}
st.plotly_chart(fig_plotly, use_container_width=True, config=plotly_config)

# Tooltip hint below the chart
st.caption("Tip: Use range buttons or drag to zoom. Double-click to reset.")

# explanatory expander under the chart
with st.expander("How to read this", expanded=False):
    st.markdown("Tip: The smoothed line is more important than the raw score. Spikes above 60 suggest elevated downside probability. Sustained readings above 80 often align with recessions and deep equity drawdowns.")

st.caption("Components: Trend • Volatility • Term structure • Breadth • Credit [+ HMM if available]")

# Acceptance checks (quick)
st.sidebar.markdown("---")
st.sidebar.markdown("**Notes**")
st.sidebar.markdown("- Components with missing data are excluded and weights renormalized per date.")
st.sidebar.markdown("- Data cached for 6h. VIX/VIX3M single-day gaps are forward-filled by 1 day.")
