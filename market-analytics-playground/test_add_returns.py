import pandas as pd
import numpy as np

from src.data_io import add_returns


def test_add_returns_basic():
    # build a minimal DataFrame with Date and Adj Close columns
    data = {
        'Date': ["2020-01-01", "2020-01-02", "2020-01-03"],
        'Adj Close': [100.0, 101.0, 102.0]
    }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])

    out = add_returns(df)

    # out should have 'ret' and two rows (first return is NaN and dropped)
    assert 'ret' in out.columns
    assert len(out) == 2

    # check numeric values (approx)
    expected = np.array([0.01, 0.00990099009900991])
    np.testing.assert_allclose(out['ret'].to_numpy(), expected, rtol=1e-6, atol=1e-12)
