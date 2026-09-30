"""Compare two result files of run_fresh.py (e.g., cloud vs local PC run of the same split) on every column except
timings: numeric columns by the maximum absolute difference, the stored thresholds (JSON) by the maximum relative
difference, and other columns as text (missing values treated as equal).
Usage: python compare_runs.py <a.csv> <b.csv> [...pairs]   -> one line per pair; exit code 1 if a rate differs"""
import json, sys, numpy as np, pandas as pd

KEY = ["alpha", "order", "method", "calib"]; SKIP = {"fit_time_full", "fit_time_foldA", "thr"}


def flat(t):
    if not isinstance(t, str): return np.array([])
    out = []
    for x in json.loads(t): out += (x if isinstance(x, list) else [x])
    return np.array([np.nan if v is None else v for v in out], float)


def compare(fa, fb):
    a, b = pd.read_csv(fa), pd.read_csv(fb)
    if len(a) != len(b): return dict(rows=(len(a), len(b)), ok=False)
    a = a.sort_values(KEY).reset_index(drop=True); b = b.sort_values(KEY).reset_index(drop=True)
    if not (a[KEY].astype(str).values == b[KEY].astype(str).values).all(): return dict(rows=len(a), ok=False, key="mismatch")
    num, txt = 0.0, []
    for c in a.columns:
        if c in SKIP: continue
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if (x.isna() != y.isna()).any(): txt.append(c); continue
            d = np.abs(x.astype(float) - y.astype(float)).max(); num = max(num, 0.0 if np.isnan(d) else float(d))
        else:
            xs = x.where(x.notna(), "").astype(str).str.lower(); ys = y.where(y.notna(), "").astype(str).str.lower()
            if (xs != ys).any(): txt.append(c)
    rel = 0.0
    for s, t in zip(a.thr, b.thr):
        u, v = flat(s), flat(t)
        if len(u) != len(v): rel = np.inf; break
        if len(u):
            same = (u == v) | (np.isnan(u) & np.isnan(v))                   # equal values, including +-inf and missing
            if not same.all(): rel = max(rel, float(np.max(np.abs(u[~same] - v[~same]) / np.maximum(1, np.abs(u[~same])))))
    return dict(rows=len(a), max_abs_rate_diff=num, max_rel_thr_diff=rel, text_cols_differing=txt, ok=(num == 0 and not txt))


if __name__ == "__main__":
    bad = 0
    for fa, fb in zip(sys.argv[1::2], sys.argv[2::2]):
        r = compare(fa, fb); bad += not r["ok"]; print(fa.split("/")[-1], r)
    sys.exit(1 if bad else 0)
