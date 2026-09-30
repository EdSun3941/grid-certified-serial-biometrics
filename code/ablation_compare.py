"""Paired comparison of ablation/sensitivity variants with the main configuration (same seeds, plain regime)."""
import glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from analyze_all import select
R = "../results"
def load(tag):
    fs = sorted(glob.glob(f"{R}/E3/{tag}_fing_x_face_s*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); d["feasible"] = d.feasible.astype(bool); return d
main = load("main"); rows = []
for tag in ["abl_sample", "abl_region", "abl_N", "sens_b001", "sens_b005", "sens_nb60", "sens_nb240"]:
    v = load(tag)
    for meth in v.method.unique():
        base_m = "LR-P1-N2" if meth.startswith("LR-P1-N") else meth
        for al in sorted(v.alpha.unique()):
            seeds = sorted(v.seed.unique())
            sv = select(v[(v.alpha == al) & (v.method == meth)], "frr_pred_gp").set_index("seed")["frr_test"]
            sm = select(main[(main.alpha == al) & (main.method == base_m) & main.seed.isin(seeds)], "frr_pred_gp").set_index("seed")["frr_test"]
            c = sv.index.intersection(sm.index)
            gv = v[(v.alpha == al) & (v.method == meth) & v.feasible]; gm = main[(main.alpha == al) & (main.method == base_m) & main.feasible & main.seed.isin(seeds)]
            p = wilcoxon(sv[c], sm[c]).pvalue if len(c) >= 5 and np.any(sv[c].values != sm[c].values) else np.nan
            rows.append({"variant": tag, "method": meth, "vs": base_m, "alpha": al, "n_seeds": len(c),
                         "frr_test_variant": sv[c].mean(), "frr_test_main": sm[c].mean(), "p_wilcoxon": p,
                         "frr_ok_train_variant": (gv.frr_train <= gv.frr_pred_gp + 1e-12).mean(),
                         "frr_ok_train_main": (gm.frr_train <= gm.frr_pred_gp + 1e-12).mean(),
                         "far_test_variant": sv.index.map(lambda s: np.nan).size and gv.groupby("seed").far_test.mean().mean(),
                         "source": f"results/E3/{tag}_fing_x_face_s*.csv vs results/E3/main_fing_x_face_s*.csv"})
t = pd.DataFrame(rows); t.to_csv(f"{R}/tables/T_ablation_paired.csv", index=False)
pd.set_option("display.width", 250); print(t.drop(columns=["source"]).round(4).to_string(index=False))
