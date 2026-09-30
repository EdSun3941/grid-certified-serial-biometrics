"""IJIS v02: constants C_s of the hyperbolic envelopes (prior GP design, fit_envelope("HYP")) of the D4 matchers on the
training halves of splits 0-9, which decide at which alpha the hyperbolic design is feasible (Sect. 6.3).
Output: results/tables/T_d4_hyp_constants.csv"""
import numpy as np, pandas as pd
from data import MATCHERS, data_file
from experiment_core import make_blocks, fit_envelope

ds = "lfw_x_fing"; D = dict(np.load(data_file(ds))); rows = []
for seed in range(10):
    tr, _ = make_blocks(D, MATCHERS[ds], seed); G = tr["_G"]
    for m in MATCHERS[ds]:
        S = tr[m][0]; env, _ = fit_envelope("HYP", S[G], S[~G], seed=seed, time_limit=60)
        (c, b), = env.terms; assert b == -1
        rows.append(dict(dataset=ds, seed=seed, matcher=m, C=c))
t = pd.DataFrame(rows); t["source"] = "d4_hyp_constants.py (fit_envelope HYP on the training halves)"
t.to_csv("../results/tables/T_d4_hyp_constants.csv", index=False); print(t.groupby("matcher").C.agg(["min", "max"]))
