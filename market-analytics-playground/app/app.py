# ============================================================
# Part 1 — Setup, Sidebar, and Data Loading
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
from collections import defaultdict

# ----------------------------
# Dark theme helpers (TrendSpider-ish)
# ----------------------------
DARK_BG  = "#0b1220"   # figure background
DARK_AX  = "#0e1525"   # axes background
GRID     = "#203049"
FG       = "#d7e3f3"   # label/tick color
LINE     = "#9ec4ff"   # main line
GREEN    = "#4caf50"
RED      = "#f44336"
NEUTRAL  = "#9e9e9e"

def styled_fig(size=(5, 3)):
    """Create a dark-styled matplotlib figure/axes."""
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID, alpha=0.6, linewidth=0.8)
    return fig, ax

# ----------------------------
# Streamlit Page setup
# ----------------------------
st.set_page_config(page_title="SPY Markov & HMM — Visual Dashboard", layout="wide")
st.title("SPY Markov Chains & Hidden Markov Models — Visual Dashboard")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Data & States")

    window_mode = st.radio(
        "History window",
        ["Full", "Last 5 years", "Last 10 years", "Custom"],
        index=1
    )

    today = date.today()
    if window_mode == "Full":
        start_date = date(1993, 1, 1); end_date = today
    elif window_mode == "Last 5 years":
        end_date = today; start_date = date(end_date.year - 5, end_date.month, end_date.day)
    elif window_mode == "Last 10 years":
        end_date = today; start_date = date(end_date.year - 10, end_date.month, end_date.day)
    else:
        start_date = st.date_input("Start date", value=date(2015, 1, 1))
        end_date   = st.date_input("End date", value=today)

    st.markdown("---")

    state_mode = st.selectbox(
        "State mode (Markov)",
        ["binary", "ternary"],
        index=0,
        help="binary = Green/Red • ternary = Green/Neutral/Red"
    )

    thr_bps = st.slider(
        "Return threshold (basis points)",
        0, 50, 10, 1,
        help="Size of a daily move to count as Green/Red. 1 bp = 0.01% (10 bps = 0.10%)."
    )
    threshold = thr_bps / 10000.0

    order = st.slider("Markov order", 1, 4, 1)
    multi_h = st.multiselect("Forecast horizons (days)", [1, 2, 3, 4, 5, 10, 20], default=[1, 2, 3, 4])

    st.markdown("---")
    st.header("HMM (optional)")
    use_hmm = st.checkbox("Enable HMM analysis", value=True)
    hmm_states = st.selectbox("# Hidden states", [2, 3], index=0)
    use_rv = st.checkbox("Include realized volatility (RV20)", value=True)
    hmm_years = st.slider("HMM train window (years)", 2, 25, 5, help="Fit on most-recent N years (saves memory).")
    bull_thresh = st.slider("Signal: Bull prob threshold", 0.5, 0.9, 0.6, 0.05)
    bear_thresh = st.slider("Signal: Bear prob threshold", 0.5, 0.9, 0.6, 0.05)

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(show_spinner=False, max_entries=2, ttl="2h")
def load_spy(start, end):
    """Download SPY and compute daily returns."""
    df = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        st.error("No SPY data from yfinance.")
        st.stop()

    # Handle multi-level column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c).strip() for c in tup if c]) for tup in df.columns.values]

    df = df.reset_index().sort_values("Date").reset_index(drop=True)

    # Find price column
    candidates = [c for c in df.columns
                  if ("adj close" in c.lower()) or (c.lower() == "close") or c.lower().endswith("_close")]
    if not candidates:
        st.error(f"Price column not found. Columns: {list(df.columns)}")
        st.stop()

    price_col = candidates[0]
    df.rename(columns={price_col: "Price"}, inplace=True)
    df["Price"] = pd.to_numeric(df["Price"], downcast="float")
    df["ret"] = df["Price"].pct_change().astype("float32")
    df = df.dropna(subset=["Price", "ret"]).reset_index(drop=True)
    return df

# Load SPY
spy = load_spy(start_date, end_date)

# ============================================================
# Part 2 — Markov Chain (discretized returns)
# ============================================================

# ---------- helpers ----------
from collections import defaultdict

def label_states(r: pd.Series, mode: str, thr: float) -> pd.Series:
    """Discretize returns into states."""
    if mode == "binary":
        lab = np.where(r > thr, "G", "R")
    else:
        lab = np.where(r > thr, "G", np.where(r < -thr, "R", "N"))
    return pd.Series(lab, index=r.index, name="state")

def k_order_transition(states: pd.Series, k: int, state_space: list[str]) -> pd.DataFrame:
    """k-order Markov transition matrix over the given discrete state series."""
    seq = states.tolist()
    counts = defaultdict(lambda: {s: 0 for s in state_space})
    for i in range(k, len(seq)):
        ctx = "-".join(seq[i - k:i])
        nxt = seq[i]
        counts[ctx][nxt] = counts[ctx].get(nxt, 0) + 1

    rows = {}
    for ctx, d in counts.items():
        tot = sum(d.values())
        rows[ctx] = {s: (d.get(s, 0) / tot if tot else 0.0) for s in state_space}

    mat = pd.DataFrame.from_dict(rows, orient="index")
    if not mat.empty:
        mat = mat.reindex(columns=state_space).sort_index()
    return mat

RENAME = {"G": "Green", "N": "Neutral", "R": "Red"}

def humanize_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    if mat.empty:
        return mat
    out = mat.copy()
    out.columns = [RENAME.get(c, c) for c in out.columns]
    out.index = ["-".join(RENAME.get(p, p) for p in idx.split("-")) for idx in out.index]
    return out

# ---------- context & captions ----------
states_series = label_states(spy["ret"], state_mode, threshold)
state_space = ["G", "R"] if state_mode == "binary" else ["G", "N", "R"]

mode_caption   = "Binary (Green/Red)" if state_mode == "binary" else "Ternary (Green/Neutral/Red)"
window_caption = f"{start_date.isoformat()} → {end_date.isoformat()}"
thr_caption    = f"Threshold: {thr_bps} bps ({threshold:.3%}) for Green"

context_raw_k   = "-".join(states_series.iloc[-order:].tolist())
context_human_k = "-".join(RENAME.get(p, p) for p in context_raw_k.split("-"))

# ---------- section ----------
st.divider()
st.markdown("## 📈 Markov Chain (discretized returns)")
st.caption(f"{mode_caption} • Window: {window_caption} • {thr_caption} • Context: `{context_human_k}`")

# Build matrix
mat = k_order_transition(states_series, order, state_space)
mat_human = humanize_matrix(mat)

# Layout
c1, c2 = st.columns([1, 1.2])

with c1:
    st.caption("Transition matrix (rows sum to 1). Index = context of last k states.")
    if not mat_human.empty:
        display_df = (mat_human * 100).round(1).reset_index().rename(columns={"index": "Context"})
        st.dataframe(
            display_df.style.format({c: "{:.1f}%" for c in display_df.columns if c != "Context"})
                             .hide(axis="index"),
            use_container_width=True
        )
    else:
        st.info("Not enough data to build this order. Try a lower order or longer window.")

with c2:
    if not mat_human.empty:
        fig, ax = styled_fig((6, 3.2))
        im = ax.imshow(mat_human.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(mat_human.columns))); ax.set_xticklabels(mat_human.columns, color=FG)
        ax.set_yticks(range(len(mat_human.index)));   ax.set_yticklabels(mat_human.index, color=FG)
        ax.set_title("Transition Heatmap (green = more likely)")
        cb = fig.colorbar(im, ax=ax)
        cb.ax.set_facecolor(DARK_AX); cb.outline.set_edgecolor(GRID)
        cb.ax.yaxis.set_tick_params(color=FG)
        for t in cb.ax.get_yticklabels(): t.set_color(FG)
        st.pyplot(fig); plt.close(fig)

# ---------- summaries ----------
if not mat.empty:
    # Most-likely next state from current context (with back-off)
    row_ctx = mat.loc[context_raw_k] if context_raw_k in mat.index else None
    used_ctx_raw = context_raw_k
    used_ctx_human = context_human_k

    if row_ctx is None and order > 1:
        for j in range(order - 1, 0, -1):
            ctx_try = "-".join(states_series.iloc[-j:].tolist())
            if ctx_try in mat.index:
                row_ctx = mat.loc[ctx_try]
                used_ctx_raw = ctx_try
                used_ctx_human = "-".join(RENAME.get(p, p) for p in ctx_try.split("-"))
                break

    if row_ctx is not None:
        last_raw   = used_ctx_raw.split("-")[-1]
        next_idx   = row_ctx.idxmax()
        next_prob  = row_ctx.max()
        cont_prob  = row_ctx.get(last_raw, 0.0)
        switch_prob = 1.0 - cont_prob

        st.markdown(
            f"**Summary:** Given `{used_ctx_human}`, most likely next day is "
            f"**{RENAME.get(next_idx, next_idx)} ({next_prob:.1%})**. "
            f"Continuation (stay {RENAME.get(last_raw, last_raw)}) = {cont_prob:.1%}; "
            f"switch = {switch_prob:.1%}."
        )

    # Strongest/weakest transitions in whole sample
    strongest = mat.stack().idxmax()
    weakest   = mat.stack().idxmin()
    st.caption(
        "Global context: strongest transition = "
        f"`{RENAME.get(strongest[0], strongest[0])}` → **{RENAME.get(strongest[1], strongest[1])}** "
        f"({mat.loc[strongest[0], strongest[1]]:.1%}); weakest = "
        f"`{RENAME.get(weakest[0], weakest[0])}` → **{RENAME.get(weakest[1], weakest[1])}** "
        f"({mat.loc[weakest[0], weakest[1]]:.1%})."
    )

    # ============================================================
    # Part 3 — One-Step & Multi-Step Forecasts
    # ============================================================

    # ---------- ONE-STEP ----------
    st.divider()
    st.markdown("## 🔮 One-Step Next-State Probabilities")
    st.caption(f"{mode_caption} • Window: {window_caption} • {thr_caption} • Context: `{context_human_k}`")

    row = mat.loc[context_raw_k] if (not mat.empty and context_raw_k in mat.index) else None
    if row is None and order > 1 and not mat.empty:
        for j in range(order - 1, 0, -1):
            ctx = "-".join(states_series.iloc[-j:].tolist())
            if ctx in mat.index:
                row = mat.loc[ctx]
                st.info(f"Back-off used: context length {j}")
                break

    if row is not None:
        row_human_pct = (row.rename(index=RENAME) * 100).round(1)
        st.dataframe(
            row_human_pct.to_frame("P(next, %)").T.style.hide(axis="index"),
            use_container_width=True
        )
        st.markdown(
            f"**Summary:** Given `{context_human_k}`, tomorrow is most likely "
            f"**{row_human_pct.idxmax()} ({row_human_pct.max():.1f}%)**."
        )

    # ---------- MULTI-STEP ----------
    st.divider()
    st.markdown("## 📊 Multi-Step Forecast (1st-Order Approximation)")
    last_state = states_series.iloc[-1]
    last_state_human = RENAME.get(last_state, last_state)
    st.caption(f"{mode_caption} • Window: {window_caption} • {thr_caption} • "
               f"Most recent state: `{last_state_human}` (1st-order baseline)")

    s1 = label_states(spy["ret"], state_mode, threshold)
    m1 = k_order_transition(s1, 1, state_space)

    if not m1.empty:
        P = m1.values
        pi = np.zeros(len(state_space))
        pi[state_space.index(s1.iloc[-1])] = 1.0

        results = []
        for h in sorted(multi_h):
            Ph = np.linalg.matrix_power(P, h)
            pi_h = pi @ Ph
            results.append({
                "Horizon (days)": h,
                **{RENAME[state_space[i]]: pi_h[i] for i in range(len(state_space))}
            })
        df_fore = pd.DataFrame(results)

        show = df_fore.copy()
        for c in show.columns:
            if c != "Horizon (days)":
                show[c] = (show[c] * 100).round(1).astype(str) + "%"

        # Layout table + chart side-by-side
        lcol, rcol = st.columns([1.25, 0.95])

        with lcol:
            st.dataframe(show.style.hide(axis="index"), use_container_width=True)

            if set(["Green", "Red"]).issubset(df_fore.columns):
                g = df_fore["Green"] * 100
                r = df_fore["Red"] * 100
                g0, g_last = g.iloc[0], g.iloc[-1]
                mean_g, mean_r = g.mean(), r.mean()
                greener = int((g > r).sum())
                redder = int((r > g).sum())
                first_h, last_h = int(df_fore["Horizon (days)"].iloc[0]), int(df_fore["Horizon (days)"].iloc[-1])

                if abs(mean_g - mean_r) < 0.25:
                    bias_txt = "overall **neutral** bias"
                elif mean_g > mean_r:
                    bias_txt = f"overall **bullish** bias (Green ≈ {mean_g:.1f}% > Red ≈ {mean_r:.1f}%)"
                else:
                    bias_txt = f"overall **bearish** bias (Red ≈ {mean_r:.1f}% > Green ≈ {mean_g:.1f}%)"

                if abs(g_last - g0) < 0.25:
                    drift_txt = "Green probability stays roughly flat"
                elif g_last > g0:
                    drift_txt = f"Green probability rises from **{g0:.1f}%** to **{g_last:.1f}%**"
                else:
                    drift_txt = f"Green probability declines from **{g0:.1f}%** to **{g_last:.1f}%**"

                st.markdown(
                    f"**Summary:** {bias_txt}. {drift_txt} over {first_h}→{last_h} days. "
                    f"Green favored in **{greener}/{len(g)}** horizons, Red in **{redder}/{len(r)}**."
                )

        with rcol:
            fig, ax = styled_fig((4.6, 2.6))
            xs = np.arange(len(df_fore));
            w = 0.28

            if "Green" in df_fore.columns:
                ax.bar(xs - w, df_fore["Green"] * 100, w, color=GREEN, edgecolor=FG, alpha=0.9, label="Green")
            if "Red" in df_fore.columns:
                ax.bar(xs, df_fore["Red"] * 100, w, color=RED, edgecolor=FG, alpha=0.9, label="Red")
            if state_mode == "ternary" and "Neutral" in df_fore.columns:
                ax.bar(xs + w, df_fore["Neutral"] * 100, w, color=NEUTRAL, edgecolor=FG, alpha=0.9, label="Neutral")

            ax.set_xticks(xs)
            ax.set_xticklabels(df_fore["Horizon (days)"].astype(int), color=FG)
            ax.set_xlabel("Horizon (days)")
            ax.set_ylabel("Probability (%)")
            ax.set_title("Next-Day State Probability by Horizon")

            leg = ax.legend(frameon=True, fontsize=8)
            leg.get_frame().set_facecolor(DARK_AX)
            leg.get_frame().set_edgecolor(GRID)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # <<< END OF PART 3 — columns are closed; Part 4 starts at root level >>>

# ============================================================
# Part 4 — Hidden Markov Model (Regime Detection) + Long-Term View
# ============================================================

st.divider()
st.markdown("## 🤖 Hidden Markov Model (Regimes)")
st.caption(
    f"Window: {start_date.isoformat()} → {end_date.isoformat()} • "
    f"Training window: last {hmm_years}y • Hidden states: {hmm_states} • "
    f"Features: ret{'+rv20' if use_rv else ''} • Bull>{bull_thresh:.0%} / Bear>{bear_thresh:.0%}"
)

# Import inside the block so the rest of the app still runs if HMM deps are missing
try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
    HMM_OK = True
except Exception:
    HMM_OK = False
    if use_hmm:
        st.warning("HMM disabled: missing dependency (install `hmmlearn` and `scikit-learn`).")

if use_hmm and HMM_OK:
    # ---------- Build features on the CURRENT data window ----------
    feats = spy[["Date", "Price", "ret"]].copy()
    if use_rv:
        feats["rv20"] = feats["ret"].rolling(20).std().bfill()
    feats = feats.dropna().reset_index(drop=True)

    # Restrict TRAINING to last N years (for speed/memory), but we'll classify broader later
    cutoff = feats["Date"].max() - pd.DateOffset(years=hmm_years)
    feats_train = feats[feats["Date"] >= cutoff].copy()

    X_cols = ["ret"] + (["rv20"] if use_rv else [])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(feats_train[X_cols].astype("float32").values)
    X_cur   = scaler.transform(feats[X_cols].astype("float32").values)

    # ---------- Fit model & infer states on the CURRENT window ----------
    try:
        hmm = GaussianHMM(n_components=hmm_states, covariance_type="full", n_iter=400, random_state=42)
        hmm.fit(X_train)
        post_cur   = hmm.predict_proba(X_cur)        # T x K
        states_idx = hmm.predict(X_cur)              # 0..K-1
        trans      = hmm.transmat_.copy()            # K x K
    except Exception as e:
        st.error(f"HMM fitting failed: {e}")
        post_cur = states_idx = trans = None

    if post_cur is not None:
        # ---------- Human regime names (by mean return) ----------
        tmp = pd.DataFrame({"state": states_idx, "ret": feats["ret"].values})
        stats_raw = tmp.groupby("state").ret.agg(["count", "mean", "std"]).rename(
            columns={"mean": "mean_daily", "std": "std_daily"}
        )

        ranks = stats_raw["mean_daily"].sort_values().index.tolist()  # ascending by mean return
        if len(ranks) == 2:
            name_map = {ranks[0]: "Bear", ranks[1]: "Bull"}
        else:  # 3+
            name_map = {ranks[0]: "Bear", ranks[-1]: "Bull"}
            for r in ranks[1:-1]:
                name_map[r] = "Neutral"

        present = list({name_map[i] for i in stats_raw.index})
        order_human = [n for n in ["Bull", "Neutral", "Bear"] if n in present]
        color_map = {"Bull": GREEN, "Neutral": NEUTRAL, "Bear": RED}

        # Reindex stats to human names & compute annualized metrics
        stats = stats_raw.copy()
        stats.index = [name_map[i] for i in stats.index]
        stats = stats.groupby(level=0).first().reindex(order_human)

        def _ann(m, s):
            r = m * 252.0
            v = s * np.sqrt(252.0)
            return pd.Series({"ann_return": r, "ann_vol": v, "sharpe": (r / v) if v > 0 else np.nan})

        stats = pd.concat([stats, stats.apply(lambda r: _ann(r["mean_daily"], r["std_daily"]), axis=1)], axis=1)

        # Reorder transition matrix & posterior to human order
        human_to_old = {v: k for k, v in name_map.items()}
        perm_old = [human_to_old[n] for n in order_human]
        trans_h = pd.DataFrame(trans[np.ix_(perm_old, perm_old)], index=order_human, columns=order_human)
        post_cur_h = pd.DataFrame(post_cur[:, perm_old], columns=order_human)

        # Most-recent regime probabilities (current window)
        last_probs = post_cur_h.iloc[-1]
        last_probs_str = " · ".join([f"{n}: {p:.0%}" for n, p in last_probs.items()])
        st.caption(f"Most-recent regime probabilities → {last_probs_str}")

        # ---------- Chart: SPY price with regime overlay (CURRENT window) ----------
        st.markdown("### SPY Price with HMM-Detected Regimes")
        fig, ax = styled_fig((9, 3.6))
        ax.plot(feats["Date"], feats["Price"], color=LINE, linewidth=1.4)

        regime_cur = post_cur_h.idxmax(axis=1)
        s0 = 0
        for i in range(1, len(regime_cur)):
            if regime_cur.iloc[i] != regime_cur.iloc[i-1]:
                ax.axvspan(feats["Date"].iloc[s0], feats["Date"].iloc[i-1],
                           color=color_map[regime_cur.iloc[i-1]], alpha=0.12)
                s0 = i
        ax.axvspan(feats["Date"].iloc[s0], feats["Date"].iloc[-1],
                   color=color_map[regime_cur.iloc[-1]], alpha=0.12)

        ax.set_xlabel("Date"); ax.set_ylabel("SPY Close"); ax.set_title("SPY Price with HMM-Detected Regimes")
        handles = [plt.Rectangle((0,0),1,1,color=color_map[n],alpha=0.4) for n in order_human]
        leg = ax.legend(handles, order_human, fontsize=8, ncols=len(order_human))
        leg.get_frame().set_facecolor(DARK_AX); leg.get_frame().set_edgecolor(GRID)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ---------- Chart: stacked regime probabilities + SPY overlay (CURRENT window) ----------
        st.markdown("### Regime Probabilities vs SPY")
        fig, ax1 = styled_fig((9, 3.6))
        ax1.stackplot(feats["Date"], [post_cur_h[c] for c in order_human],
                      labels=order_human, colors=[color_map[c] for c in order_human],
                      alpha=0.85, edgecolor=GRID, linewidth=0.3)
        ax1.set_ylabel("P(state)"); ax1.set_ylim(0, 1.0); ax1.set_xlabel("Date")

        ax2 = ax1.twinx(); ax2.set_facecolor("none")
        ax2.plot(feats["Date"], feats["Price"], color=LINE, linewidth=1.2, alpha=0.9)
        ax2.set_ylabel("SPY Close"); ax2.tick_params(colors=FG); ax2.yaxis.set_label_coords(1.06, 0.5)
        leg = ax1.legend(fontsize=8, ncols=len(order_human), loc="upper left")
        leg.get_frame().set_facecolor(DARK_AX); leg.get_frame().set_edgecolor(GRID)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ---------- Long-term view: classify FULL history (≈1995 → today) ----------
        @st.cache_data(show_spinner=False, ttl="6h")
        def _load_spy_full(_end):
            # Always fetch full history (from 1993) for the long-term view
            return load_spy(date(1993, 1, 1), _end)

        spy_full = _load_spy_full(end_date)

        feats_full = spy_full[["Date", "Price", "ret"]].copy()
        if use_rv:
            feats_full["rv20"] = feats_full["ret"].rolling(20).std().bfill()
        feats_full = feats_full.dropna().reset_index(drop=True)

        X_full_hist = scaler.transform(feats_full[X_cols].astype("float32").values)
        post_full   = hmm.predict_proba(X_full_hist)
        post_full_h = pd.DataFrame(post_full[:, perm_old], columns=order_human)
        regime_full = post_full_h.idxmax(axis=1)

        st.markdown("### SPY Price with HMM-Detected Regimes — Full History")
        fig, ax = styled_fig((11, 3.8))
        ax.plot(feats_full["Date"], feats_full["Price"], color=LINE, linewidth=1.2)

        s0 = 0
        for i in range(1, len(regime_full)):
            if regime_full.iloc[i] != regime_full.iloc[i-1]:
                ax.axvspan(feats_full["Date"].iloc[s0], feats_full["Date"].iloc[i-1],
                           color=color_map[regime_full.iloc[i-1]], alpha=0.12)
                s0 = i
        ax.axvspan(feats_full["Date"].iloc[s0], feats_full["Date"].iloc[-1],
                   color=color_map[regime_full.iloc[-1]], alpha=0.12)

        # Focus from ~1995 while keeping earlier data for context if present
        try:
            ax.set_xlim(pd.Timestamp("1995-01-01"), feats_full["Date"].iloc[-1])
        except Exception:
            pass

        ax.set_xlabel("Date"); ax.set_ylabel("SPY Close")
        ax.set_title("Long-Term Regimes (1995 → today)")
        handles = [plt.Rectangle((0,0),1,1,color=color_map[n],alpha=0.4) for n in order_human]
        leg = ax.legend(handles, order_human, fontsize=8, ncols=len(order_human))
        leg.get_frame().set_facecolor(DARK_AX); leg.get_frame().set_edgecolor(GRID)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.caption(
            "Model **trained on last "
            f"{hmm_years} years**, but regimes **inferred across full history**. "
            "Use this for long-cycle context while keeping training fast."
        )

        # ---------- Tables: transition + expected duration ----------
        tm = (trans_h * 100).round(1)
        st.caption("HMM transition matrix (daily transition probabilities)")
        st.dataframe(tm.rename_axis("From → To").style.format("{:.1f}%"), use_container_width=True)

        # Expected duration in days: 1 / (1 - Pii)
        persistence = {}
        for s in order_human:
            pii = float(trans_h.loc[s, s])
            persistence[s] = np.inf if (1 - pii) <= 1e-9 else 1.0 / (1.0 - pii)
        persist_txt = ", ".join([f"{k}: ~{(v if v < 5e4 else float('inf')):.0f} days"
                                 for k, v in sorted(persistence.items(), key=lambda x: x[1], reverse=True)])
        st.markdown(f"**Transition takeaway:** Regimes are persistent. Expected duration — {persist_txt}.")

        st.markdown("---")

        # ---------- Table: per-state performance (human-friendly) ----------
        stats_h = stats.copy().loc[order_human].rename(columns={
            "count":"Days in regime", "mean_daily":"Avg daily return", "std_daily":"Daily volatility",
            "ann_return":"Annualized return", "ann_vol":"Annualized volatility", "sharpe":"Sharpe"
        })
        cols = ["Days in regime","Avg daily return","Daily volatility","Annualized return","Annualized volatility","Sharpe"]
        show_stats = stats_h[cols].copy()
        for c in ["Avg daily return","Daily volatility","Annualized return","Annualized volatility"]:
            show_stats[c] = (show_stats[c]*100).map(lambda x: f"{x:.2f}%")
        show_stats["Sharpe"] = show_stats["Sharpe"].map(lambda x: f"{x:.2f}")
        st.caption("Per-state behavior (returns & risk, daily → annualized)")
        st.dataframe(show_stats, use_container_width=True)

        best = stats_h["Annualized return"].idxmax()
        worst = stats_h["Annualized return"].idxmin()
        best_ret = stats_h.loc[best, "Annualized return"] * 100
        worst_ret = stats_h.loc[worst, "Annualized return"] * 100
        best_sh = stats_h.loc[best, "Sharpe"]; worst_sh = stats_h.loc[worst, "Sharpe"]
        st.markdown(
            f"**Performance takeaway:** **{best}** regime shows strongest growth "
            f"(*{best_ret:.1f}%/yr*, Sharpe *{best_sh:.2f}*).  "
            f"**{worst}** is weakest "
            f"(*{abs(worst_ret):.1f}%/yr loss*, Sharpe *{worst_sh:.2f}*).  "
            f"Use regime probabilities to tilt exposure."
        )

        st.markdown(
            "**Column guide:**  \n"
            "- **Avg daily return** – average daily return in that regime.  \n"
            "- **Daily volatility** – stdev of daily returns.  \n"
            "- **Annualized return/volatility** – daily metrics scaled to yearly (×252, ×√252).  \n"
            "- **Sharpe** – risk-adjusted return (higher is better).  \n"
            "- **Days in regime** – number of trading days historically in each regime."
        )

        # ---------- Strategies: definitions + equity curves (with selectable window) ----------
        if "Bull" in post_cur_h.columns and "Bear" in post_cur_h.columns:
            st.markdown("### Strategy backtests (based on current-window regimes)")
            st.markdown(
                "**Strategy definitions:**  \n"
                "• 🟩 **Long/Neutral:** hold SPY in **Bull** regime; stay in cash otherwise (flat line when out).  \n"
                "• 🟥 **Long/Short:** long in **Bull**, short in **Bear** (neutral when neither).  \n"
                "• ⚪ **Buy & Hold:** always long SPY (benchmark)."
            )

            # Choose performance window without touching the main data window
            perf_choice = st.selectbox(
                "Performance window for strategies",
                ["Use current data window", "Last 5 years", "Last 10 years", "Last 15 years", "Last 20 years"],
                index=0,
                help="This only affects the equity curves & table below."
            )
            years_map = {
                "Last 5 years": 5,
                "Last 10 years": 10,
                "Last 15 years": 15,
                "Last 20 years": 20
            }
            if perf_choice == "Use current data window":
                feats_perf = feats.copy()
                post_perf_h = post_cur_h.copy()
            else:
                ny = years_map[perf_choice]
                cutoff_perf = feats["Date"].max() - pd.DateOffset(years=ny)
                mask = feats["Date"] >= cutoff_perf
                feats_perf = feats.loc[mask].reset_index(drop=True)
                post_perf_h = post_cur_h.loc[mask].reset_index(drop=True)

            # Signals & equity curves for the chosen window
            bull_p = post_perf_h.get("Bull", pd.Series(0, index=feats_perf.index)).values
            bear_p = post_perf_h.get("Bear", pd.Series(0, index=feats_perf.index)).values

            long_neutral = (bull_p > bull_thresh).astype(int)
            long_short   = np.where(bull_p > bull_thresh, 1, np.where(bear_p > bear_thresh, -1, 0))

            eq_ln = (1 + feats_perf["ret"].values * long_neutral).cumprod()
            eq_ls = (1 + feats_perf["ret"].values * long_short).cumprod()
            bh    = (1 + feats_perf["ret"].values).cumprod()

            # --- Benchmarked summary (includes Buy & Hold) ---
            strategies = ["Buy & Hold", "Long/Neutral", "Long/Short"]
            finals     = [bh[-1], eq_ln[-1], eq_ls[-1]]
            returns    = [v - 1.0 for v in finals]

            bh_final = finals[0]
            bh_ret   = returns[0]

            summary_rows = []
            for name, v, r in zip(strategies, finals, returns):
                delta_pp = (r - bh_ret) * 100.0  # percentage points vs benchmark
                summary_rows.append({
                    "Strategy": name,
                    "Final equity ($)": v,
                    "Total return (%)": r * 100.0,
                    "Δ vs Buy & Hold (pp)": delta_pp
                })
            df_summary = pd.DataFrame(summary_rows)

            # Keep Buy & Hold first; sort others by Final equity desc
            df_summary = pd.concat([
                df_summary[df_summary["Strategy"] == "Buy & Hold"],
                df_summary[df_summary["Strategy"] != "Buy & Hold"].sort_values("Final equity ($)", ascending=False)
            ], ignore_index=True)

            # Nicely formatted table
            fmt = {"Final equity ($)": "{:.2f}", "Total return (%)": "{:.1f}%", "Δ vs Buy & Hold (pp)": "{:+.1f}pp"}
            st.caption("Final equity values (starting from $1). Δ = out/underperformance vs benchmark over the same window.")
            st.dataframe(df_summary.style.format(fmt), use_container_width=True)

            # One-line takeaway
            leader = df_summary.iloc[1:].sort_values("Final equity ($)", ascending=False).iloc[0]
            st.markdown(
                f"**Takeaway:** Best strategy vs benchmark here is **{leader['Strategy']}** "
                f"({leader['Δ vs Buy & Hold (pp)']:+.1f}pp), finishing at **${leader['Final equity ($)']:.2f}**."
            )

            # Equity plots
            c5, c6, c7 = st.columns(3)
            for vec, title, col in [
                (eq_ln, "Equity: Long/Neutral", c5),
                (eq_ls, "Equity: Long/Short",   c6),
                (bh,    "Equity: Buy & Hold",   c7),
            ]:
                with col:
                    fig, ax = styled_fig((4.6, 2.8))
                    ax.plot(feats_perf["Date"], vec, linewidth=1.3, color=LINE)
                    ax.set_xlabel("Date"); ax.set_ylabel("Equity (Growth of $1)")
                    ax.set_title(title, fontsize=10)
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.markdown(
                "📊 **Interpretation:** Each line shows the growth of a \\$1 investment using the chosen regime strategy.  "
                "Values above 1.0 mean profits; below 1.0 indicate losses.  "
                "A flat line in **Long/Neutral** means the strategy is in cash (no exposure) outside Bull regimes."
            )