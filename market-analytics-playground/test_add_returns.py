import sys
from pathlib import Path
proj = Path(__file__).resolve().parents[0]
if str(proj) not in sys.path:
    sys.path.insert(0, str(proj))

import traceback
try:
    import src.data_io as dio
    print('Loaded module:', dio.__file__)
    cache = proj / 'data' / 'raw' / 'spy.feather'
    print('Cache path:', cache)
    print('Exists:', cache.exists())
    df = dio.load_prices(str(cache))
    print('Columns:', df.columns.tolist())
    # show normalized map
    def _norm(s: str):
        import re
        s2 = str(s).lower()
        s2 = re.sub(r"[^a-z0-9]", " ", s2)
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2
    for c in df.columns:
        print('  ', c, '->', _norm(c))
    # call add_returns
    df2 = dio.add_returns(df)
    print('add_returns succeeded, head:')
    print(df2.head())
except Exception:
    traceback.print_exc()

