"""Empirical check of Proposition 1 (per-stage conservativeness): single-stage GP designs, training data.
Corner-point dominance (main) vs sample-point dominance (abl_sample)."""
import glob, pandas as pd
rows = []
for tag, pat in [("corner (main)", "../results/E3/main_*.csv"), ("sample points (ablation)", "../results/E3/abl_sample_*.csv")]:
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))], ignore_index=True)
    d = d[(d.kind == "GP") & (d.feasible == True) & (d.n_stages == 1)]
    if tag.startswith("sample"): d = d[d.method.isin(["LR-P1-N2", "MONO", "LR-P1-N2-rel"])]
    d["ok"] = d.frr_train <= d.frr_pred_gp + 1e-12; d["excess"] = (d.frr_train - d.frr_pred_gp).clip(lower=0)
    for (ds, m), g in d.groupby(["dataset", "method"]):
        rows.append({"dominance": tag, "dataset": ds, "method": m, "n_designs": len(g), "frac_realized_le_pred": g.ok.mean(),
                     "max_excess": g.excess.max(), "source": pat})
t = pd.DataFrame(rows); t.to_csv("../results/tables/T_prop1_check.csv", index=False)
print(t.groupby("dominance").agg(n=("n_designs", "sum"), ok_min=("frac_realized_le_pred", "min")))
