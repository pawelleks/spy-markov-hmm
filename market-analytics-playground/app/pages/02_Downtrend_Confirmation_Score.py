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
    try:
        # if hb_raw is list/array same length as df, align by order
        hb_arr = np.asarray(hb_raw, dtype=float)
        if len(hb_arr) == len(df):
            hmm_bear = pd.Series(hb_arr, index=df.index).astype("float32")
        else:
            # try to interpret as (date, val) pairs or dict
            hb_df = pd.DataFrame(hb_raw)
            if {"Date", "Value"}.issubset(hb_df.columns):
                hb_df["Date"] = pd.to_datetime(hb_df["Date"]).dt.normalize()
                tmp = hb_df.set_index("Date")["Value"].reindex(df["Date"]).ffill()
                hmm_bear = tmp.astype("float32")
    except Exception:
        hmm_bear = None

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

# Component table for latest date
comp_table = []
# Human-friendly names for signals
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

if len(last_idx):
    i = last
    total_weight = 0.0
    total_contrib = 0.0
    for j, name in enumerate(comp_names):
        rawv = sig_vals[i, j]
        sc = np.nan if np.isnan(rawv) else float(rawv)
        w = float(W[i, j])
        contrib = np.nan if np.isnan(rawv) else sc * w
        total_weight += w
        total_contrib += 0.0 if np.isnan(contrib) else contrib
        comp_table.append({
            "Signal": human_names.get(name, name),
            "Value/raw": ("NA" if np.isnan(rawv) else f"{sc:.2f}"),
            "Score (0-1)": ("NA" if np.isnan(rawv) else f"{sc:.2f}"),
            "Weight": f"{w:.2f}",
            "Weighted contribution": ("NA" if np.isnan(contrib) else f"{contrib:.2f}")
        })
    # Append a totals row (SUM) showing total weight and total weighted contribution
    comp_table.append({
        "Signal": "SUM",
        "Value/raw": "",
        "Score (0-1)": "",
        "Weight": f"{total_weight:.2f}",
        "Weighted contribution": f"{total_contrib:.2f}"
    })
    df_comp_table = pd.DataFrame(comp_table)

    # --- Enhanced UI for Component contributions (presentation only) ---
    # User controls
    st.sidebar.markdown("")
    show_explanations = st.sidebar.checkbox("Show explanations", value=True, key="dcs_show_explanations")
    show_sparklines = st.sidebar.checkbox("Show sparklines", value=False, key="dcs_show_sparklines")
    group_by_category = st.sidebar.checkbox("Group by category", value=True, key="dcs_group_by_category")
    # compact / view controls (define early so rendering below can reference them)
    compact_mode = st.sidebar.checkbox("Compact mode (recommended)", value=True, key="dcs_compact_mode")
    show_sparklines_compact = st.sidebar.checkbox("Show sparklines (compact)", value=False, key="dcs_show_sparklines_compact")
    # effective sparklines flag used in rendering
    try:
        show_sparklines_effective = show_sparklines or show_sparklines_compact
    except NameError:
        show_sparklines_effective = show_sparklines_compact
    view_mode = st.sidebar.radio("View", ["Grid", "Table (dense)"], index=0, key="dcs_view_mode")

    # Category mapping
    category_map = {
        "price_lt_ema50": "Trend",
        "ema20_lt_ema50": "Trend",
        "mom21_lt_0": "Trend",
        "atr_gt_sma63": "Volatility",
        "rv20_gt_rv63": "Volatility",
        "vix_term_pos": "Market Structure",
        "rsp_spy_63_neg": "Breadth",
        "hyg_lqd_21_neg": "Credit",
        "hmm_bear_prob": "State Models"
    }

    # Explanations
    explanations = {
        "Price < EMA(50)": "Price trading below its 50-day EMA — bearish trend.",
        "EMA(20) < EMA(50)": "Short-term EMA below medium-term EMA — trend weakness.",
        "21-day Momentum < 0": "Negative 21-day price change — downward momentum.",
        "ATR(14)/Price > 63-day SMA": "Rising ATR relative to its SMA — volatility pickup.",
        "Realized Vol (20d) > (63d)": "Short-term realized vol higher than longer-term — expanding variance.",
        "VIX − VIX3M > 0 (term)": "Positive VIX term slope — near-term volatility premium.",
        "RSP/SPY 63d return < 0 (breadth)": "Equal-weight underperforming cap-weighted — weak breadth.",
        "HYG/LQD 21d return < 0 (credit)": "High-yield underperforms investment-grade — credit stress.",
        "HMM: Bear probability": "Model-estimated probability of a bear regime (0-1)."
    }

    # helpers: color interpolation
    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def rgb_to_hex(r, g, b):
        return f'#{int(r):02x}{int(g):02x}{int(b):02x}'
    green = hex_to_rgb('1e7f54')
    amber = hex_to_rgb('b08b00')
    redc = hex_to_rgb('b33a3a')

    def interp_color(v):
        # v between 0..1 -> interpolate green->amber->red
        if np.isnan(v):
            return 'transparent'
        v = float(np.clip(v, 0.0, 1.0))
        if v <= 0.5:
            t = v / 0.5
            r = green[0] + (amber[0]-green[0]) * t
            g = green[1] + (amber[1]-green[1]) * t
            b = green[2] + (amber[2]-green[2]) * t
        else:
            t = (v-0.5)/0.5
            r = amber[0] + (redc[0]-amber[0]) * t
            g = amber[1] + (redc[1]-amber[1]) * t
            b = amber[2] + (redc[2]-amber[2]) * t
        # return rgba with some alpha suited for dark bg
        return f'rgba({int(r)},{int(g)},{int(b)},0.18)'

    # status badge
    def status_badge(v):
        if np.isnan(v):
            return ('NA', NEUTRAL)
        if v <= 0.33:
            return ('OK', '#1e7f54')
        if v <= 0.66:
            return ('CAUTION', '#b08b00')
        return ('RISK', '#b33a3a')

    # Sparkline generator
    import io, base64
    def make_sparkline(series: pd.Series, color='#00d4ff', width=120, height=32):
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.plot(series.index, series.values, color=color, linewidth=1.2)
        ax.fill_between(series.index, series.values, alpha=0.04, color=color)
        ax.axis('off')
        plt.margins(x=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return buf

    # Build a structured list with category and optional sparkline series
    rows = []
    # map internal keys to underlying series for sparklines
    for r in comp_table[:-1]:  # exclude SUM here
        key = None
        # try to find the component key by matching human name to internal map
        # reverse map
        rev = {v:k for k,v in human_names.items()}
        internal = rev.get(r['Signal'], None)
        if internal is None:
            # fallback: try exact matches
            for k in comp_names:
                if human_names.get(k,'') == r['Signal']:
                    internal = k; break
        cat = category_map.get(internal, 'Misc')
        # base series selection
        spark_series = None
        try:
            if internal in ['price_lt_ema50', 'ema20_lt_ema50']:
                spark_series = (signals[internal].astype(float)).fillna(0).iloc[-60:]
            elif internal == 'mom21_lt_0':
                # momentum scaled 0..1
                tmp = mom21.copy().astype('float')
                tmp = (tmp - tmp.rolling(252, min_periods=1).min()) / (tmp.rolling(252, min_periods=1).max() - tmp.rolling(252, min_periods=1).min() + 1e-9)
                spark_series = tmp.iloc[-60:].fillna(0)
            elif internal == 'atr_gt_sma63':
                tmp = (atr14 / spy_price) / (atr63 / spy_price.replace(0, np.nan) + 1e-9)
                spark_series = tmp.iloc[-60:].replace([np.inf,-np.inf],np.nan).fillna(method='ffill').fillna(0)
            elif internal == 'rv20_gt_rv63':
                tmp = (rv20 / (rv63 + 1e-9))
                spark_series = tmp.iloc[-60:].replace([np.inf,-np.inf],np.nan).fillna(method='ffill').fillna(0)
            elif internal == 'vix_term_pos':
                tmp = term_diff.fillna(0)
                # scale to 0..1 by percentile
                spark_series = ((tmp - tmp.min()) / (tmp.max()-tmp.min()+1e-9)).iloc[-60:].fillna(0)
            elif internal == 'rsp_spy_63_neg':
                tmp = ratio_return63.fillna(0)
                # negative returns -> positive signal; scale abs(return)
                spark_series = ((-tmp).clip(lower=0)).iloc[-60:].fillna(0)
            elif internal == 'hyg_lqd_21_neg':
                tmp = credit_ratio21.fillna(0)
                spark_series = ((-tmp).clip(lower=0)).iloc[-60:].fillna(0)
            elif internal == 'hmm_bear_prob' and hmm_bear is not None:
                spark_series = hmm_bear.iloc[-60:].fillna(0)
        except Exception:
            spark_series = None

        rows.append({
            'internal': internal,
            'Signal': r['Signal'],
            'Category': cat,
            'Value/raw': r['Value/raw'],
            'Score': (np.nan if r['Score (0-1)']=="NA" else float(r['Score (0-1)'])),
            'Weight': float(r['Weight']) if r['Weight']!='' else 0.0,
            'Contribution': (np.nan if r['Weighted contribution']=="NA" or r['Weighted contribution']=='' else float(r['Weighted contribution'])),
            'spark': spark_series
        })

    # Compute model confidence
    available = [rr for rr in rows if not (rr['Score'] is np.nan)]
    active_count = sum(1 for rr in available if rr['Score'] is not None and rr['Score'] >= 0.5)
    total_available = len([rr for rr in rows if rr['Score'] is not np.nan])
    if active_count <= 2:
        conf = ('Low', GREEN)
    elif active_count <= 5:
        conf = ('Medium', '#b08b00')
    else:
        conf = ('High', RED)

    # Model confidence box
    col1, col2, col3 = st.columns([1,1,3])
    with col1:
        st.markdown(f"**Model confidence:** <span style='color:{conf[1]}'>{conf[0]}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Active bearish signals:** <span style='color:{FG}'>{active_count} / {len([r for r in rows if r['Score'] is not np.nan])}</span>", unsafe_allow_html=True)
    with col3:
        st.caption("Signals with Score ≥ 0.5 counted as active bearish signals.")

    # Render table: grouped or flat
    def render_row(rr):
        # create columns: Signal | Sparkline | Value | Score | Weight | Contribution | Status
        cols = st.columns([2, 1, 1, 1, 1, 1, 1])
        # Signal + explanation
        with cols[0]:
            st.markdown(f"**{rr['Signal']}**")
            if show_explanations:
                expl = explanations.get(rr['Signal'], '')
                if expl:
                    st.caption(expl)
        # sparkline
        with cols[1]:
            if show_sparklines and rr['spark'] is not None and len(rr['spark']):
                buf = make_sparkline(rr['spark'])
                st.image(buf, use_column_width=True)
        # Value
        with cols[2]:
            st.markdown(f"<div style='text-align:right;color:{FG};font-size:12px'>{rr['Value/raw']}</div>", unsafe_allow_html=True)
        # Score with heatmap background
        score_bg = interp_color(rr['Score'] if rr['Score'] is not None else np.nan)
        with cols[3]:
            val = 'NA' if rr['Score'] is np.nan else f"{rr['Score']:.2f}"
            st.markdown(f"<div style='background:{score_bg};padding:6px;border-radius:4px;text-align:right;color:{FG};font-size:12px'>{val}</div>", unsafe_allow_html=True)
        # Weight
        with cols[4]:
            st.markdown(f"<div style='text-align:right;color:{FG};font-size:12px'>{rr['Weight']:.2f}</div>", unsafe_allow_html=True)
        # Contribution with heatmap
        contrib_bg = interp_color((rr['Contribution'] - 0.0) if rr['Contribution'] is not np.nan else np.nan)
        with cols[5]:
            val = 'NA' if rr['Contribution'] is np.nan else f"{rr['Contribution']:.2f}"
            st.markdown(f"<div style='background:{contrib_bg};padding:6px;border-radius:4px;text-align:right;color:{FG};font-size:12px'>{val}</div>", unsafe_allow_html=True)
        # Status badge
        stat, color = status_badge(rr['Score'] if rr['Score'] is not None else np.nan)
        with cols[6]:
            st.markdown(f"<div style='background:{color};color:{DARK_BG};padding:6px;border-radius:4px;text-align:center;font-weight:bold'>{stat}</div>", unsafe_allow_html=True)

    if group_by_category:
        # group by category
        grouped = {}
        for rr in rows:
            grouped.setdefault(rr['Category'], []).append(rr)
        for cat, items in grouped.items():
            # compact_mode: render inline dense grid (no expander) to save vertical space
            # robustly handle items that may use 'score' or 'Score' keys and 'weight' or 'Weight'
            def _get_num(it, keys):
                for k in keys:
                    if isinstance(it, dict) and k in it:
                        try:
                            v = it[k]
                            return float(v) if v is not None and (not (isinstance(v, str) and v.strip()=='')) else np.nan
                        except Exception:
                            return np.nan
            def get_field(it, keys, default=''):
                for k in keys:
                    if isinstance(it, dict) and k in it:
                        return it[k]
                return default

            active_count = sum(1 for it in items if not np.isnan(_get_num(it, ('score','Score'))) and _get_num(it, ('score','Score'))>=0.5)
            weight_sum = sum((_get_num(it, ('weight','Weight')) if not np.isnan(_get_num(it, ('weight','Weight'))) else 0.0) for it in items)
            cat_title = f"{cat} — {active_count}/{len(items)} bearish • sum(w)={weight_sum:.2f}"
            if compact_mode:
                # small header
                st.markdown(f"<div style='margin-top:6px;margin-bottom:4px;color:{FG};font-size:13px;font-weight:600'>{cat_title}</div>", unsafe_allow_html=True)
                # render tight grid: choose up to 4 cols to minimize rows
                ncols = 4
                chunks = [items[i:i+ncols] for i in range(0, len(items), ncols)]
                for chunk in chunks:
                    cols = st.columns(ncols, gap='small')
                    for col, it in zip(cols, chunk):
                        with col:
                            # compact card: one-line title, tiny spark, thin bar, weight+status on same line
                            title = get_field(it, ('signal','Signal'))
                            tooltip = explanations.get(title, '') if 'explanations' in locals() else ''
                            # sparkline (very small)
                            if show_sparklines_effective and it['spark'] is not None:
                                try:
                                    buf = make_sparkline(it['spark'], width=60, height=22)
                                    st.image(buf, use_column_width=True)
                                except Exception:
                                    pass
                            # compact HTML card
                            pct = 0 if np.isnan(it['Score']) else it['Score']
                            pct_norm = np.clip(pct/100.0 if pct>1 else pct, 0, 1)
                            stat_txt, stat_col = status_badge(get_field(it, ('Score','score'), np.nan) if not np.isnan(get_field(it, ('Score','score'), np.nan)) else np.nan)
                            card_html = (
                                f"<div class='mini-card' style='height:56px;overflow:hidden;'>"
                                f"<div class='title' title='{tooltip}' style='font-size:12px;margin-bottom:4px'>{title}</div>"
                                f"<div class='mini-bar'><div class='fill' style='width:{pct_norm*100:.1f}%'></div></div>"
                                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-top:6px;font-size:11px;color:{FG}'>"
                                f"<span style='opacity:0.9'>w={it['Weight']:.2f}</span>"
                                f"<span style='background:{stat_col};color:{DARK_BG};padding:2px 6px;border-radius:4px;font-weight:700'>{stat_txt}</span>"
                                f"</div></div>"
                            )
                            st.markdown(card_html, unsafe_allow_html=True)
            else:
                # non-compact: keep expanders (less dense, more details)
                expanded_default = not compact_mode
                with st.expander(cat_title, expanded=expanded_default):
                    ncols = 3
                    chunks = [items[i:i+ncols] for i in range(0, len(items), ncols)]
                    for chunk in chunks:
                        cols = st.columns(ncols, gap='small')
                        for col, it in zip(cols, chunk):
                            with col:
                                score_pct = 0.0 if np.isnan(it['score']) else it['score']
                                stat_txt, stat_col = status_badge(it['score'] if not np.isnan(it['score']) else np.nan)
                                title = it['signal']
                                tooltip = explanations.get(it['signal'], '') if 'explanations' in locals() else ''
                                if show_sparklines_effective and it['spark'] is not None:
                                    try:
                                        buf = make_sparkline(it['spark'], width=80, height=28)
                                        st.image(buf, use_column_width=True)
                                    except Exception:
                                        pass
                                st.markdown(f"<div class='mini-card'><div class='title' title='{tooltip}'>{title}</div>", unsafe_allow_html=True)
                                pct = 0 if np.isnan(score_pct) else score_pct
                                pct_norm = np.clip(pct/100.0 if pct>1 else pct, 0, 1)
                                bar_html = f"<div class='mini-bar'><div class='fill' style='width:{pct_norm*100:.1f}%'></div></div>"
                                st.markdown(bar_html, unsafe_allow_html=True)
                                st.markdown(f"<div class='meta' style='display:flex;justify-content:space-between;align-items:center;margin-top:4px'><span>w={it['weight']:.2f}</span><span style='color:{FG}'>{'NA' if np.isnan(it['score']) else f'{it['score']:.2f}'}</span></div></div>", unsafe_allow_html=True)
    else:
        # flat list
        for rr in rows:
            render_row(rr)

    # Totals row (SUM) rendered at bottom
    st.markdown("---")
    st.markdown(f"**Totals — Weight:** {total_weight:.2f} — **Weighted contribution:** {total_contrib:.2f}")

    # --- Compact & Dense UI (grid of mini-cards + dense table view) ---
    # Sidebar controls were defined earlier above; reuse them here.
    st.sidebar.caption("Compact mode reduces padding, font sizes, and uses a grid of mini-cards for faster scanning.")
    # view_mode is declared earlier; do not redeclare to avoid duplicate widget keys

    # Build rows (simple representation)
    rows = []
    rev = {v: k for k, v in human_names.items()}
    for r in comp_table[:-1]:
        internal = rev.get(r['Signal'])
        if internal is None:
            for k in comp_names:
                if human_names.get(k, '') == r['Signal']:
                    internal = k; break
        cat = category_map.get(internal, 'Misc')
        score_val = (np.nan if r['Score (0-1)'] == 'NA' else float(r['Score (0-1)']))
        weight_val = (0.0 if r['Weight'] == '' else float(r['Weight']))
        contrib_val = (np.nan if r['Weighted contribution'] in ("NA", "") else float(r['Weighted contribution']))
        # try to build small spark series
        spark_series = None
        try:
            if internal in ['price_lt_ema50', 'ema20_lt_ema50']:
                spark_series = signals[internal].fillna(0).iloc[-40:]
            elif internal == 'mom21_lt_0':
                tmp = mom21.copy().astype(float)
                tmp = (tmp - tmp.rolling(252, min_periods=1).min()) / (tmp.rolling(252, min_periods=1).max() - tmp.rolling(252, min_periods=1).min() + 1e-9)
                spark_series = tmp.iloc[-40:].fillna(0)
            elif internal == 'atr_gt_sma63':
                tmp = (atr14 / spy_price) / (atr63 / spy_price.replace(0, np.nan) + 1e-9)
                spark_series = tmp.iloc[-40:].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
            elif internal == 'rv20_gt_rv63':
                tmp = (rv20 / (rv63 + 1e-9))
                spark_series = tmp.iloc[-40:].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
            elif internal == 'vix_term_pos' and term_diff is not None:
                tmp = term_diff.fillna(0)
                spark_series = ((tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-9)).iloc[-40:].fillna(0)
            elif internal == 'rsp_spy_63_neg' and ratio_return63 is not None:
                tmp = ratio_return63.fillna(0); spark_series = ((-tmp).clip(lower=0)).iloc[-40:].fillna(0)
            elif internal == 'hyg_lqd_21_neg' and credit_ratio21 is not None:
                tmp = credit_ratio21.fillna(0); spark_series = ((-tmp).clip(lower=0)).iloc[-40:].fillna(0)
            elif internal == 'hmm_bear_prob' and hmm_bear is not None:
                spark_series = hmm_bear.iloc[-40:].fillna(0)
        except Exception:
            spark_series = None

        rows.append({
            'internal': internal,
            'signal': r['Signal'],
            'category': cat,
            'score': score_val,
            'weight': weight_val,
            'contrib': contrib_val,
            'spark': spark_series,
            'raw': r['Value/raw']
        })

    # summarise
    available = [rr for rr in rows if not np.isnan(rr['score'])]
    active_count = sum(1 for rr in available if rr['score'] >= 0.5)
    total_available = len(available)
    if active_count <= 2:
        conf_txt, conf_col = ('Low', GREEN)
    elif active_count <= 5:
        conf_txt, conf_col = ('Medium', '#b08b00')
    else:
        conf_txt, conf_col = ('High', RED)

    # top summary strip (single line)
    s1, s2, s3 = st.columns([1, 1, 1])
    s1.markdown(f"**Confidence:** <span style='color:{conf_col}'>{conf_txt}</span>", unsafe_allow_html=True)
    s2.markdown(f"**Active bearish signals:** <span style='color:{FG}'>{active_count} / {total_available}</span>", unsafe_allow_html=True)
    score_display = f"{current_score:.1f}" if not np.isnan(current_score) else "N/A"
    s3.markdown(f"**Score:** <span style='color:{FG}'>{score_display}</span>", unsafe_allow_html=True)

    # compact CSS: denser mini-cards and tighter expanders for single-screen fit
    if compact_mode:
        st.markdown("""
        <style>
        /* Mini-cards: reduced padding & margin, smaller fonts */
        .mini-card { background: #0f1724; padding:4px; border-radius:6px; margin:2px; }
        .mini-card .title { font-size:11px; color: #d7e3f3; font-weight:600; line-height:1; }
        .mini-card .meta { font-size:10px; color: #bfcddb; }
        .mini-bar { height:8px; background:#0f1a28; border-radius:6px; margin-top:6px }
        .mini-bar .fill { height:100%; background: linear-gradient(90deg, rgba(158,196,255,1), rgba(0,200,255,0.6)); border-radius:6px; }
        /* Tighter expander summary and caption */
        .streamlit-expanderHeader { padding: 4px 8px; }
        /* Reduce padding inside st.columns content */
        .css-1lcbmhc.e1fqkh3o { padding: 2px 4px !important; }
        </style>
        """, unsafe_allow_html=True)

    if view_mode == 'Table (dense)':
        df_table = pd.DataFrame([{
            'Signal': r['signal'],
            'Score': (np.nan if np.isnan(r['score']) else round(r['score'],2)),
            'Weight': round(r['weight'],2),
            'Contribution': (np.nan if np.isnan(r['contrib']) else round(r['contrib'],2)),
            'Status': status_badge(r['score'])[0]
        } for r in rows])
        # style and render
        # Inject dense table CSS so rows are compact and fit on one screen
        st.markdown("""
        <style>
        /* Target st.dataframe table cells */
        [data-testid="stDataFrame"] table {
            font-size:12px;
        }
        [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td {
            padding: 4px 6px !important;
            line-height: 1.1 !important;
        }
        /* Reduce widget container padding that can add vertical space */
        .css-1v0mbdj.e1tzin5v { padding: 4px 8px !important; }
        </style>
        """, unsafe_allow_html=True)

        st.dataframe(df_table.style.hide_index().format({'Score':'{:.2f}','Weight':'{:.2f}','Contribution':'{:.2f}'}), use_container_width=True, height=min(520, 32 + 26 * len(df_table)))
    else:
        # Grid mode
        grouped = {}
        for rr in rows:
            grouped.setdefault(rr['category'], []).append(rr)
        for cat, items in grouped.items():
            # compact_mode: render inline dense grid (no expander) to save vertical space
            cat_title = f"{cat} — {sum(1 for it in items if not np.isnan(it['score']) and it['score']>=0.5)}/{len(items)} bearish • sum(w)={sum(it['weight'] for it in items if not np.isnan(it['weight'])):.2f}"
            if compact_mode:
                # small header
                st.markdown(f"<div style='margin-top:6px;margin-bottom:4px;color:{FG};font-size:13px;font-weight:600'>{cat_title}</div>", unsafe_allow_html=True)
                # render tight grid: choose up to 4 cols to minimize rows
                ncols = 4
                chunks = [items[i:i+ncols] for i in range(0, len(items), ncols)]
                for chunk in chunks:
                    cols = st.columns(ncols, gap='small')
                    for col, it in zip(cols, chunk):
                        with col:
                            # compact card: one-line title, tiny spark, thin bar, weight+status on same line
                            title = get_field(it, ('signal','Signal'))
                            tooltip = explanations.get(title, '') if 'explanations' in locals() else ''
                            # sparkline (very small)
                            if show_sparklines_effective and it['spark'] is not None:
                                try:
                                    buf = make_sparkline(it['spark'], width=60, height=22)
                                    st.image(buf, use_column_width=True)
                                except Exception:
                                    pass
                            # compact HTML card
                            pct = 0 if np.isnan(it['Score']) else it['Score']
                            pct_norm = np.clip(pct/100.0 if pct>1 else pct, 0, 1)
                            stat_txt, stat_col = status_badge(get_field(it, ('Score','score'), np.nan) if not np.isnan(get_field(it, ('Score','score'), np.nan)) else np.nan)
                            card_html = (
                                f"<div class='mini-card' style='height:56px;overflow:hidden;'>"
                                f"<div class='title' title='{tooltip}' style='font-size:12px;margin-bottom:4px'>{title}</div>"
                                f"<div class='mini-bar'><div class='fill' style='width:{pct_norm*100:.1f}%'></div></div>"
                                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-top:6px;font-size:11px;color:{FG}'>"
                                f"<span style='opacity:0.9'>w={it['Weight']:.2f}</span>"
                                f"<span style='background:{stat_col};color:{DARK_BG};padding:2px 6px;border-radius:4px;font-weight:700'>{stat_txt}</span>"
                                f"</div></div>"
                            )
                            st.markdown(card_html, unsafe_allow_html=True)
            else:
                # non-compact: keep expanders (less dense, more details)
                expanded_default = not compact_mode
                with st.expander(cat_title, expanded=expanded_default):
                    ncols = 3
                    chunks = [items[i:i+ncols] for i in range(0, len(items), ncols)]
                    for chunk in chunks:
                        cols = st.columns(ncols, gap='small')
                        for col, it in zip(cols, chunk):
                            with col:
                                score_pct = 0.0 if np.isnan(it['score']) else it['score']
                                stat_txt, stat_col = status_badge(it['score'] if not np.isnan(it['score']) else np.nan)
                                title = it['signal']
                                tooltip = explanations.get(it['signal'], '') if 'explanations' in locals() else ''
                                if show_sparklines_effective and it['spark'] is not None:
                                    try:
                                        buf = make_sparkline(it['spark'], width=80, height=28)
                                        st.image(buf, use_column_width=True)
                                    except Exception:
                                        pass
                                st.markdown(f"<div class='mini-card'><div class='title' title='{tooltip}'>{title}</div>", unsafe_allow_html=True)
                                pct = 0 if np.isnan(score_pct) else score_pct
                                pct_norm = np.clip(pct/100.0 if pct>1 else pct, 0, 1)
                                bar_html = f"<div class='mini-bar'><div class='fill' style='width:{pct_norm*100:.1f}%'></div></div>"
                                st.markdown(bar_html, unsafe_allow_html=True)
                                st.markdown(f"<div class='meta' style='display:flex;justify-content:space-between;align-items:center;margin-top:4px'><span>w={it['weight']:.2f}</span><span style='color:{FG}'>{'NA' if np.isnan(it['score']) else f'{it['score']:.2f}'}</span></div></div>", unsafe_allow_html=True)
    # End of compact/table rendering

else:
    st.info("No valid scores available for the selected window.")

# Charts
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
