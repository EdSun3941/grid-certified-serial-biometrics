"""IJIS v08 (review M6, minor 8): D4 with candidate sets of the proposed design restricted to the face matcher alone,
to the face and one finger, and to all three matchers (three-stage chains), each with the same subject-bootstrap
calibration and the same selection by training FRR (ties averaged) as in the paper; paired comparison (corrected
resampled t-test, 10 splits) of each candidate set and of the SPRT and LLR fusion with the face-only design.
Output: results/tables/T_d4_subsets.csv"""
import glob, numpy as np, pandas as pd
from analyze_rev import corrected_t
R = "../results"; OUT = f"{R}/tables"
SETS = {"face only": lambda o: o == "face_S",
        "face and one finger": lambda o: o.count(">") == 1 and "face_S" in o,
        "three matchers": lambda o: o.count(">") == 2,
        "all orders (paper)": lambda o: True}


def main():
    fs = sorted(glob.glob(f"{R}/E3fresh/fresh_lfw_x_fing_s*.csv")); df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df = df[(df.calib == "boot") & (df.feasible == True)].copy()
    par = df.kind == "parallel"; df.loc[par, "stages_gen"] = df.loc[par, "n_stages"].astype(float)   # parallel fusion acquires all
    rows = []
    for (a, seed), g in df.groupby(["alpha", "seed"]):
        gp = g[g.method == "LR-P1-N2"]
        for name, f in SETS.items():
            c = gp[gp.order.map(f)]
            if len(c) == 0: continue
            t = c[np.isclose(c.frr_train, c.frr_train.min(), rtol=0, atol=1e-12)]
            rows.append(dict(alpha=a, seed=seed, design=name, frr_test=t.frr_test.mean(), far_ok=float((t.far_test <= a).mean()),
                             stages_gen=t.stages_gen.mean()))
        for m in ["S3-SPRT", "P1-LLR"]:
            c = g[g.method == m]
            t = c[np.isclose(c.frr_train, c.frr_train.min(), rtol=0, atol=1e-12)]
            rows.append(dict(alpha=a, seed=seed, design=m, frr_test=t.frr_test.mean(), far_ok=float((t.far_test <= a).mean()),
                             stages_gen=t.stages_gen.mean()))
    sel = pd.DataFrame(rows); out = []
    for (a, d), g in sel.groupby(["alpha", "design"]):
        ref = sel[(sel.alpha == a) & (sel.design == "face only")].set_index("seed").frr_test
        x = g.set_index("seed").frr_test; c = ref.index.intersection(x.index); diff = (x[c] - ref[c]).values
        row = dict(alpha=a, design=d, n=len(g), frr_test=g.frr_test.mean(), n_met=g.far_ok.sum(), stages_gen=g.stages_gen.mean())
        if d != "face only":
            m, ci, p = corrected_t(diff); row.update(diff_vs_face=m, ci_lo=ci[0], ci_hi=ci[1], p=p, n_lower=int((diff < 0).sum()), n_higher=int((diff > 0).sum()))
        out.append(row)
    T = pd.DataFrame(out); T["source"] = f"results/E3fresh/fresh_lfw_x_fing_s*.csv ({len(fs)} files), calibration boot"
    T.to_csv(f"{OUT}/T_d4_subsets.csv", index=False); print(T.drop(columns="source").round(4).to_string())


if __name__ == "__main__":
    main()
