"""Per-design decomposition of the predicted system FRR (revision): for every uncalibrated proposed design
(LR-P1-N2, all orders and splits) the four successive differences are computed per design and summarized by
median and interquartile range.  Output: results/tables/T_rev_decomposition.csv"""
import glob, numpy as np, pandas as pd
d = pd.concat([pd.read_csv(f) for f in glob.glob("../results/E3/main_*.csv")], ignore_index=True)
d = d[(d.method == "LR-P1-N2") & (d.feasible == True)].copy()
steps = [("series bound", "frr_pred_gp", "frr_pred_exact"), ("envelope", "frr_pred_exact", "frr_emp_model"),
         ("dependence", "frr_emp_model", "frr_train"), ("generalization", "frr_train", "frr_test")]
rows = []
for (ds, a), g in d.groupby(["dataset", "alpha"]):
    r = {"dataset": ds, "alpha": a, "n_designs": len(g)}
    for c in ["frr_pred_gp", "frr_pred_exact", "frr_emp_model", "frr_train", "frr_test"]: r[c + "_median"] = g[c].median()
    for name, a0, a1 in steps:
        dd = g[a1] - g[a0]
        r[f"{name}_median"] = dd.median(); r[f"{name}_q25"] = dd.quantile(0.25); r[f"{name}_q75"] = dd.quantile(0.75)
    rows.append(r)
t = pd.DataFrame(rows); t["source"] = "results/E3/main_*.csv (LR-P1-N2, uncalibrated, all orders)"
t.to_csv("../results/tables/T_rev_decomposition.csv", index=False)
print(t.drop(columns=["source"]).round(4).T.to_string())
