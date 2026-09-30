"""Re-run only the MILP rows of E7v2 after fixing the coefficient re-solve (scaled columns)."""
import sys, time, numpy as np, pandas as pd, glob
from experiment_core import fitting_data, make_blocks
from envelope_solvers import fit_milp_p2
from data import data_file
for f in sorted(glob.glob("../results/E7v2/e7v2_*.csv")):
    d = pd.read_csv(f); ds, m, w = d.dataset.iloc[0], d.matcher.iloc[0], d.which.iloc[0]
    D = dict(np.load(data_file(ds))); tr, te = make_blocks(D, [m], 0); del D; G = tr["_G"]; S = tr[m][0]
    for idx in d.index[d.method == "MILP (HiGHS)"]:
        r = d.loc[idx]; nb = int(r.nbins) if "nbins" in d and pd.notna(r.get("nbins", np.nan)) else 120
        x, y, _, _ = fitting_data(S[G], S[~G], nbins=nb)
        B = np.round(np.arange(-3, 1e-9, r.bstep), 4) if ("bstep" in d and pd.notna(r.get("bstep", np.nan))) else np.round(np.arange(-3, 1e-9, 0.02), 4)
        t = time.time(); _, v, info = fit_milp_p2(x, y, B, int(r.N), time_limit=600)
        d.loc[idx, ["value", "time", "complete"]] = [v, time.time() - t, info["complete"]]
    d.to_csv(f, index=False); print(f, flush=True)
