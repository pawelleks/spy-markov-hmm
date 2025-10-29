# SPY Markov Chains & Hidden Markov Model Dashboard

**Live app:** https://spy-markov-hmm.streamlit.app

This dashboard analyzes SPY using:
- Discretized Markov chains (binary/tri-state) with k-order transitions
- Hidden Markov Model (HMM) regimes (probabilities, transitions, persistence)
- Strategy backtests (Long/Neutral, Long/Short) vs Buy & Hold
- Flexible windows, thresholds, and clean visuals

## Run locally
```bash
pip install -r requirements.txt
streamlit run market-analytics-playground/app/app.py
