"""Analysis of the fresh-seed replication and of subset D4 (results/E3fresh, run_fresh.py).
Selection per (dataset, seed, alpha, method, calibration): smallest design-data FRR (training half for `boot`, fold A for
`xfit`), ties averaged, exactly as in analyze_rev.py.  Paired differences to the proposed design: corrected resampled
t-test (Nadeau-Bengio, K = number of splits, n_test/n_train = 1) and Holm within each (dataset, alpha, calibration).
Outputs results/tables/T_fresh_selected.csv, T_fresh_system.csv, T_fresh_stats.csv, T_fresh_compliance.csv."""
import glob, numpy as np, pandas as pd
from analyze_rev import corrected_t, holm, MAIN

R = "../results"; OUT = f"{R}/tables"
PRIMARY = ["HYP", "S1-Marcialis", "S2-Symmetric", "S4-Direct", "S3-SPRT", "P0-Parallel", "P1-LLR", "P2-LogReg"]


def select(df):
    rows = []
    f = df[df.feasible == True]
    for key, g in f.groupby(["dataset", "seed", "alpha", "method", "calib"]):
        crit = "frr_train" if key[4] == "boot" else "frr_foldA"
        mn = g[crit].min(); t = g[np.isclose(g[crit], mn, rtol=0, atol=1e-12)]; a = key[2]
        rows.append(dict(zip(["dataset", "seed", "alpha", "method", "calib"], key), n_tied=len(t), frr_test=t.frr_test.mean(),
                         far_test=t.far_test.mean(), far_ok=float((t.far_test <= a).mean()), stages_gen=t.stages_gen.mean(),
                         stages_imp=t.stages_imp.mean(), orders="|".join(t.order.astype(str))))
    return pd.DataFrame(rows)


def main():
    fs = sorted(glob.glob(f"{R}/E3fresh/fresh_*.csv")); df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    src = f"results/E3fresh/fresh_*.csv ({len(fs)} files)"
    sel = select(df); sel["source"] = src; sel.to_csv(f"{OUT}/T_fresh_selected.csv", index=False)
    nseeds = sel.groupby(["dataset", "calib"]).seed.nunique()
    agg = sel.groupby(["dataset", "alpha", "calib", "method"]).agg(n_seeds=("seed", "size"), frr_test=("frr_test", "mean"),
                                                                    frr_test_sd=("frr_test", "std"), far_test=("far_test", "mean"),
                                                                    far_ok=("far_ok", "mean"), stages_gen=("stages_gen", "mean"),
                                                                    stages_imp=("stages_imp", "mean")).reset_index()
    agg["far_over_alpha"] = agg.far_test / agg.alpha; agg["source"] = src
    agg.to_csv(f"{OUT}/T_fresh_system.csv", index=False)
    st = []
    for (ds, a, cal), g in sel.groupby(["dataset", "alpha", "calib"]):
        mv = g[g.method == MAIN].set_index("seed").frr_test
        for m in PRIMARY:
            o = g[g.method == m].set_index("seed").frr_test; c = mv.index.intersection(o.index)
            if len(c) < 5: st.append(dict(dataset=ds, alpha=a, calib=cal, method=m, n=len(c))); continue
            d = (mv[c] - o[c]).values; mean, ci, p = corrected_t(d)
            st.append(dict(dataset=ds, alpha=a, calib=cal, method=m, n=len(c), mean_diff=mean, ci_lo=ci[0], ci_hi=ci[1],
                           rel_diff=mean / o[c].mean() if o[c].mean() > 0 else np.nan, p_corr_t=p,
                           n_better=int((d < 0).sum()), n_worse=int((d > 0).sum())))
    st = pd.DataFrame(st); st["p_holm"] = np.nan
    for col in ["mean_diff", "ci_lo", "ci_hi", "rel_diff", "p_corr_t", "n_better", "n_worse"]:
        if col not in st: st[col] = np.nan
    for _, g in st[st.p_corr_t.notna()].groupby(["dataset", "alpha", "calib"]):
        st.loc[g.index, "p_holm"] = holm(g.p_corr_t.values)
    st["source"] = src; st.to_csv(f"{OUT}/T_fresh_stats.csv", index=False)
    # compliance of the deployed (selected) proposed designs: original ten splits (E3b) + fresh splits
    old = pd.read_csv(f"{OUT}/T_rev_selected.csv"); old = old[old.method == MAIN].assign(calib="boot", split_set="original 0-9")
    new = sel[sel.method == MAIN].assign(split_set=lambda x: np.where(x.dataset == "lfw_x_fing", "D4 0-9", "fresh"))
    comp = pd.concat([old[["dataset", "seed", "alpha", "calib", "far_ok", "far_test", "frr_test", "split_set"]],
                      new[["dataset", "seed", "alpha", "calib", "far_ok", "far_test", "frr_test", "split_set"]]], ignore_index=True)
    cagg = comp.groupby(["dataset", "alpha", "calib", "split_set"]).agg(n=("seed", "size"), far_ok=("far_ok", "mean"),
                                                                        far_over_alpha=("far_test", "mean"), frr_test=("frr_test", "mean")).reset_index()
    cagg["far_over_alpha"] = cagg.far_over_alpha / cagg.alpha; cagg["source"] = f"T_rev_selected.csv + {src}"
    cagg.to_csv(f"{OUT}/T_fresh_compliance.csv", index=False)
    print(nseeds.to_string())
    print(agg[agg.calib == "boot"].pivot_table(index=["dataset", "alpha"], columns="method", values="frr_test").round(4).to_string())
    print(agg[agg.calib == "xfit"].pivot_table(index=["dataset", "alpha"], columns="method", values="frr_test").round(4).to_string())
    print(cagg.round(3).to_string())
    print(st[["dataset", "alpha", "calib", "method", "n", "mean_diff", "ci_lo", "ci_hi", "p_holm", "n_better", "n_worse"]].round(4).to_string())


if __name__ == "__main__":
    main()
