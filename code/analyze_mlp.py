"""IJIS v04: analysis of the MLP fusion baseline (results/E3mlp, run_mlp.py), a secondary comparison outside the primary
Holm family.  For each subset, split set (original splits 0-9 of D1-D3; new splits 10-29 of D1-D3; splits 0-9 of D4),
calibration and alpha: mean test FRR, FAR compliance and mean test FAR / alpha of the MLP, and paired differences of
the MLP to the proposed design and to logistic-regression fusion (selected designs of T_rev_selected.csv or
T_fresh_selected.csv), with the corrected resampled t-test (unadjusted p and 95% CI).
Outputs results/tables/T_mlp_system.csv and T_mlp_stats.csv."""
import glob, numpy as np, pandas as pd
from analyze_rev import corrected_t, MAIN

R = "../results"; OUT = f"{R}/tables"


def split_set(ds, seed):
    return "D4 0-9" if ds == "lfw_x_fing" else ("original 0-9" if seed < 10 else "fresh")


def main():
    fs = sorted(glob.glob(f"{R}/E3mlp/mlp_*.csv")); m = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    m["split_set"] = [split_set(d, s) for d, s in zip(m.dataset, m.seed)]
    m["far_ok"] = (m.far_test <= m.alpha).astype(float).where(m.feasible == True)
    src = f"results/E3mlp/mlp_*.csv ({len(fs)} files)"
    sysr = m[m.feasible == True].groupby(["dataset", "split_set", "calib", "alpha"]).agg(
        n=("seed", "size"), frr_test=("frr_test", "mean"), far_ok=("far_ok", "mean"), far_test=("far_test", "mean")).reset_index()
    nall = m.groupby(["dataset", "split_set", "calib", "alpha"]).seed.size().rename("n_all").reset_index()
    sysr = sysr.merge(nall, on=["dataset", "split_set", "calib", "alpha"]); sysr["far_over_alpha"] = sysr.far_test / sysr.alpha
    sysr["source"] = src; sysr.to_csv(f"{OUT}/T_mlp_system.csv", index=False)
    old = pd.read_csv(f"{OUT}/T_rev_selected.csv").assign(calib="boot")
    new = pd.read_csv(f"{OUT}/T_fresh_selected.csv")
    ref = pd.concat([old[["dataset", "seed", "alpha", "method", "calib", "frr_test"]], new[["dataset", "seed", "alpha", "method", "calib", "frr_test"]]],
                    ignore_index=True)
    rows = []
    for (ds, ss, cal, a), g in m[m.feasible == True].groupby(["dataset", "split_set", "calib", "alpha"]):
        mv = g.set_index("seed").frr_test
        for other in [MAIN, "P2-LogReg", "P1-LLR"]:
            o = ref[(ref.dataset == ds) & (ref.calib == cal) & np.isclose(ref.alpha, a) & (ref.method == other)].set_index("seed").frr_test
            c = mv.index.intersection(o.index)
            if len(c) < 5: continue
            d = (mv[c] - o[c]).values; mean, ci, p = corrected_t(d)
            rows.append(dict(dataset=ds, split_set=ss, calib=cal, alpha=a, other=other, n=len(c), mean_diff=mean, ci_lo=ci[0], ci_hi=ci[1],
                             p_unadj=p, n_mlp_lower=int((d < 0).sum()), n_mlp_higher=int((d > 0).sum()),
                             frr_mlp=mv[c].mean(), frr_other=o[c].mean()))
    st = pd.DataFrame(rows); st["source"] = f"{src}; T_rev_selected.csv, T_fresh_selected.csv; corrected resampled t-test (unadjusted)"
    st.to_csv(f"{OUT}/T_mlp_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(sysr.round(4).to_string())
    print(st[["dataset", "split_set", "calib", "alpha", "other", "n", "frr_mlp", "frr_other", "mean_diff", "ci_lo", "ci_hi", "p_unadj",
              "n_mlp_lower", "n_mlp_higher"]].round(4).to_string())


if __name__ == "__main__":
    main()
