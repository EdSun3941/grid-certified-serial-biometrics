"""Round-2 check (reviewer 2, N2): the envelopes behind the system-level results (E1, fitted by the subgradient
LR-BB of the first version) are refitted with the exact-dual LR-BB of Sec. IV (same constraint generation, same
training halves) and compared.  Output: results/E1xd/xd_<dataset>_s<seed>.csv (per envelope)."""
import json, os, sys, numpy as np, pandas as pd
from experiment_core import make_blocks, fit_envelope
from envelope_solvers import env_eval
from data import MATCHERS, data_file, roc, staircase_points
ds, seed = sys.argv[1], int(sys.argv[2])
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None     # optional subset of methods (merged into the file)
e1 = pd.read_csv(f"../results/E1/main_{ds}_s{seed}.csv")
D = dict(np.load(data_file(ds))); M = MATCHERS[ds]; tr, _ = make_blocks(D, M, seed); del D; G = tr["_G"]
rows = []; os.makedirs("../results/E1xd", exist_ok=True)
for meth in (ONLY or ["LR-P1-N2", "LR-P1-N3", "LR-P2-N2", "LR-P1-N2-rel", "LR-P2-N2-rel"]):
    for m in M:
        r = e1[(e1.method == meth) & (e1.matcher == m)]
        if len(r) == 0: continue
        r = r.iloc[0]; old = [tuple(t) for t in json.loads(r.terms)]
        S = tr[m][0]; env, info = fit_envelope("XD-" + meth[3:], S[G], S[~G], seed=seed, time_limit=300)
        _, far, frr = roc(np.sort(S[G]), np.sort(S[~G])); xd, yd = staircase_points(far, frr, "corner")
        k = (xd >= env.xmin) & (xd <= env.xmax); xd, yd = xd[k], yd[k]
        go, gn = env_eval(old, xd), env(xd)
        rows.append(dict(dataset=ds, seed=seed, matcher=m, method=meth, terms_old=r.terms, terms_new=json.dumps(env.terms),
                         sse_old=float(np.sum((go - yd) ** 2)), sse_new=float(np.sum((gn - yd) ** 2)),
                         max_rel_curve_diff=float(np.max(np.abs(gn - go) / go)), rounds_new=info["rounds"], rounds_old=r.rounds,
                         n_fit_new=info["n_fit"], n_fit_old=r.n_fit, gap_new=info.get("gap", np.nan), complete_new=info.get("complete", np.nan),
                         dominates_new=info["dominates_all"]))
        print(rows[-1]["method"], m, rows[-1]["sse_old"], rows[-1]["sse_new"], rows[-1]["max_rel_curve_diff"], flush=True)
out = pd.DataFrame(rows); f = f"../results/E1xd/xd_{ds}_s{seed}.csv"
if ONLY and os.path.exists(f):
    prev = pd.read_csv(f); out = pd.concat([prev[~prev.method.isin(ONLY)], out], ignore_index=True)
out.to_csv(f, index=False)
