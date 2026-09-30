"""Revision: FAR compliance and FRR-prediction validity of all feasible proposed designs under the
subject-bootstrap calibration (E3b), by chain type, and of the SELECTED designs.
Output: results/tables/T_rev_conservativeness.csv"""
import glob, numpy as np, pandas as pd
from analyze_all import chain_type
b = pd.concat([pd.read_csv(f) for f in glob.glob("../results/E3b/main_*.csv")], ignore_index=True)
b = b[(b.method == "LR-P1-N2") & (b.feasible == True)].copy(); b["chain"] = b.order.map(chain_type)
rows = []
for (ds, a), g in b.groupby(["dataset", "alpha"]):
    for ch, h in list(g.groupby("chain")) + [("all", g)]:
        rows.append({"dataset": ds, "alpha": a, "chain": ch, "n": len(h), "far_ok_train": float((h.far_train <= a).mean()),
                     "far_ok_test": float((h.far_test <= a).mean()), "far_ratio_median": float((h.far_test / a).median()),
                     "frr_ok_test": float((h.frr_test <= h.frr_pred_cal).mean())})
sel = pd.read_csv("../results/tables/T_rev_selected.csv"); s = sel[sel.method == "LR-P1-N2"].copy()
s["chains"] = s.orders.map(lambda o: "|".join(sorted(set(chain_type(x) for x in o.split("|")))))
for (ds, a), g in s.groupby(["dataset", "alpha"]):
    rows.append({"dataset": ds, "alpha": a, "chain": "selected", "n": len(g), "far_ok_test": g.far_ok.mean(),
                 "far_ratio_median": float((g.far_test / a).median()), "frr_ok_test": g.pred_ok.mean(),
                 "selected_chain_types": ";".join(sorted(set(g.chains))),
                 "selected_n_stages": float(g.n_stages.mean())})
t = pd.DataFrame(rows); t["source"] = "results/E3b/main_*.csv (LR-P1-N2); T_rev_selected.csv"
t.to_csv("../results/tables/T_rev_conservativeness.csv", index=False)
print(t.drop(columns=["source"]).round(3).to_string())
