"""Supplementary: SPRT baseline (S3) on D1, all orders with >=2 stages."""
import sys, os, numpy as np, pandas as pd
from experiment_core import make_blocks, all_orders
from serial_design import rule_sprt, simulate
from data import MATCHERS, data_file
ds, seed = sys.argv[1], int(sys.argv[2])
D = dict(np.load(data_file(ds))); M = MATCHERS[ds]; tr, te = make_blocks(D, M, seed); del D
G, Gt = tr["_G"], te["_G"]; rows = []
alphas = [1e-2, 1e-3] if ds == "fing_x_face" else [1e-2, 1e-3, 1e-4]
for alpha in alphas:
    for order in [o for o in all_orders(M) if len(o) >= 2]:
        out = rule_sprt([tr[m][0] for m in order], G, alpha)
        r = {"dataset": ds, "seed": seed, "alpha": alpha, "order": ">".join(order), "n_stages": len(order), "method": "S3-SPRT"}
        if out is None: r["feasible"] = False
        else:
            thr, llrs = out
            cum_tr, cum_te = [], []; a1 = a2 = 0
            for s, m in enumerate(order):
                a1 = a1 + llrs[s](tr[m][0]); cum_tr.append(a1); a2 = a2 + llrs[s](te[m][0]); cum_te.append(a2)
            fa, fr, _, _ = simulate(cum_tr, G, thr); r.update(feasible=True, far_train=fa, frr_train=fr)
            fa, fr, sg, si = simulate(cum_te, Gt, thr); r.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si)
        rows.append(r)
os.makedirs("../results/S3", exist_ok=True)
pd.DataFrame(rows).to_csv(f"../results/S3/sprt_{ds}_s{seed}.csv", index=False)
print("done", ds, seed)
