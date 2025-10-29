import numpy as np, pandas as pd
from collections import defaultdict

def label_binary(r: pd.Series, thr: float = 0.0) -> pd.Series:
    return pd.Series(np.where(r > thr, "G", "R"), index=r.index, name="state")

def label_ternary(r: pd.Series, thr: float = 0.001) -> pd.Series:
    lab = np.where(r > thr, "G", np.where(r < -thr, "R", "N"))
    return pd.Series(lab, index=r.index, name="state")

def k_order_transition(states: pd.Series, k: int, state_space: list[str]) -> pd.DataFrame:
    seq = states.tolist()
    counts = defaultdict(lambda: {s:0 for s in state_space})
    for i in range(k, len(seq)):
        ctx = "-".join(seq[i-k:i]); nxt = seq[i]
        counts[ctx][nxt] = counts[ctx].get(nxt, 0) + 1
    rows = {}
    for ctx, d in counts.items():
        total = sum(d.values())
        rows[ctx] = {s: (d.get(s,0)/total if total else 0.0) for s in state_space}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()