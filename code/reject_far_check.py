"""Where do the reject thresholds of the proposed GP designs lie on the FAR axis?
Re-solves the uncalibrated GP for every multi-stage order (LR-P1-N2 envelopes from results/E1)
and records the stage FAR at each reject threshold.  Output: results/tables/T_reject_far.csv"""
import json, numpy as np, pandas as pd
from experiment_core import all_orders
from serial_design import Envelope, gp_design
from data import MATCHERS

rows = []
for ds, alphas in [("fing_x_face", [1e-2, 1e-3]), ("fing_x_fing", [1e-2, 1e-3, 1e-4]), ("face_x_face", [1e-2, 1e-3, 1e-4])]:
    M = MATCHERS[ds]
    for seed in range(10):
        e1 = pd.read_csv(f"../results/E1/main_{ds}_s{seed}.csv")
        envs = {}
        for m in M:
            r = e1[(e1.matcher == m) & (e1.method == "LR-P1-N2")].iloc[0]
            envs[m] = Envelope("posy", [tuple(t) for t in json.loads(r.terms)], r.xmin, r.xmax)
        for alpha in alphas:
            for order in all_orders(M):
                if len(order) < 2: continue
                d = gp_design([envs[m] for m in order], alpha)
                if d is None: continue
                for s, st in enumerate(d["stages"][:-1]):
                    rows.append({"dataset": ds, "seed": seed, "alpha": alpha, "order": ">".join(order), "stage": s + 1,
                                 "a_acc": st["a_acc"], "a_rej": st["a_rej"]})
d = pd.DataFrame(rows)
t = d.groupby(["dataset", "alpha"]).agg(n=("a_rej", "size"), a_rej_median=("a_rej", "median"),
        frac_a_rej_gt_0p1=("a_rej", lambda v: float(np.mean(v > 0.1))), a_acc_median=("a_acc", "median")).reset_index()
t["source"] = "results/E1/main_*.csv (LR-P1-N2 terms); GP re-solved by reject_far_check.py"
t.to_csv("../results/tables/T_reject_far.csv", index=False)
print(t.to_string(index=False))
