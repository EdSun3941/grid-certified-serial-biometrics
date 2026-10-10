"""Analysis of the fresh-seed replication and of subset D4 (results/E3fresh, run_fresh.py).
Selection per (dataset, seed, alpha, method, calibration): smallest design-data FRR (training half for `boot`, fold A for
`xfit`), ties averaged, exactly as in analyze_rev.py.  v08: for `xfit` the order is fixed on fold A before fold B is used;
a fold-B calibration failure leaves the split without a deployed design (see select()).  Paired differences to the proposed design: corrected resampled
t-test (Nadeau-Bengio, K = number of splits, n_test/n_train = 1) and Holm within each (dataset, alpha, calibration).
v12 (review of the v11 manuscript, R1): one estimand throughout.  The tie among the best fold-A orders is broken uniformly at
random before fold B is used, so split j deploys a design with probability p_j (p_deploy).  Deployment counts and the
counts of designs meeting alpha are expected values (sums of p_j and p_j * far_ok_j); FRR, FAR and stage counts of the
deployed designs are the corresponding deployment-conditional means sum_j p_j r_j / sum_j p_j (previously the equal-weight
mean of r_j over the splits with p_j > 0).  Paired comparisons: the two methods break their ties independently, the
comparison is conditional on both designs being deployed, and split j has weight p_j^A p_j^B (corrected_t_w).  For the
paper calibration (boot) every p_j = 1, so its results are unchanged.
Outputs results/tables/T_fresh_selected.csv, T_fresh_system.csv, T_fresh_stats.csv, T_fresh_compliance.csv."""
import glob, numpy as np, pandas as pd
from analyze_rev import corrected_t, corrected_t_w, holm, MAIN

R = "../results"; OUT = f"{R}/tables"
PRIMARY = ["HYP", "S1-Marcialis", "S2-Symmetric", "S4-Direct", "S3-SPRT", "P0-Parallel", "P1-LLR", "P2-LogReg"]


def select(df):
    """boot: smallest training-half FRR among the feasible (calibrated) designs, ties averaged.
    xfit (v08, corrected protocol): the order is fixed on fold A alone -- smallest fold-A FRR among the designs that
    are feasible on fold A (frr_foldA present), ties averaged as a uniform random tie-break fixed before fold B is
    used.  The fold-B recalibration of the final threshold is then applied to that design only; if it fails, no
    design is deployed for that split (no reselection).  p_deploy is the share of tied fold-A designs that could be
    deployed; the test rates are means over the deployed ones (conditional on deployment)."""
    rows = []
    keys = ["dataset", "seed", "alpha", "method", "calib"]
    for key, g in df.groupby(keys):
        a = key[2]
        if key[4] == "boot":
            f = g[g.feasible == True]
            if len(f) == 0: continue
            mn = f.frr_train.min(); t = f[np.isclose(f.frr_train, mn, rtol=0, atol=1e-12)]; dep = t
        else:
            f = g[g.frr_foldA.notna()]
            if len(f) == 0:
                rows.append(dict(zip(keys, key), n_tied=0, n_deployed=0, p_deploy=0.0, design_foldA=False)); continue
            mn = f.frr_foldA.min(); t = f[np.isclose(f.frr_foldA, mn, rtol=0, atol=1e-12)]; dep = t[t.feasible == True]
        row = dict(zip(keys, key), n_tied=len(t), n_deployed=len(dep), p_deploy=len(dep) / len(t), design_foldA=True,
                   orders="|".join(t.order.astype(str)), orders_deployed="|".join(dep.order.astype(str)))
        if len(dep):
            row.update(frr_test=dep.frr_test.mean(), far_test=dep.far_test.mean(), far_ok=float((dep.far_test <= a).mean()),
                       stages_gen=dep.stages_gen.mean(), stages_imp=dep.stages_imp.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    fs = sorted(glob.glob(f"{R}/E3fresh/fresh_*.csv")); df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    # files written before the v05 fix of run_fresh.py record one stage for parallel fusion, which acquires every modality
    par = df.kind == "parallel"; df.loc[par, "stages_gen"] = df.loc[par, "n_stages"].astype(float); df.loc[par, "stages_imp"] = df.loc[par, "n_stages"].astype(float)
    src = f"results/E3fresh/fresh_*.csv ({len(fs)} files)"
    sel = select(df); sel["source"] = src; sel.to_csv(f"{OUT}/T_fresh_selected.csv", index=False); sel = sel.drop(columns="source")
    nseeds = sel.groupby(["dataset", "calib"]).seed.nunique()
    sel["met"] = sel.p_deploy * sel.far_ok.fillna(0)            # deployed and test FAR <= alpha (tie-averaged)
    dep_ = sel.groupby(["dataset", "alpha", "calib", "method"]).agg(n_splits=("seed", "size"), n_deployed=("p_deploy", "sum"),
                                                                     n_met=("met", "sum")).reset_index()
    def wmean(g):                                                # v12: deployment-conditional means (weights p_j)
        w = g.p_deploy; sw = w.sum(); m = (w * g.frr_test).sum() / sw
        sd = np.sqrt((w * (g.frr_test - m) ** 2).sum() / (sw - (w ** 2).sum() / sw)) if len(g) > 1 and sw - (w ** 2).sum() / sw > 0 else np.nan
        return pd.Series(dict(n_seeds=len(g), frr_test=m, frr_test_sd=sd, far_test=(w * g.far_test).sum() / sw,
                              stages_gen=(w * g.stages_gen).sum() / sw, stages_imp=(w * g.stages_imp).sum() / sw))
    agg = sel[sel.p_deploy > 0].groupby(["dataset", "alpha", "calib", "method"]).apply(wmean, include_groups=False).reset_index()
    agg = dep_.merge(agg, on=["dataset", "alpha", "calib", "method"], how="left")
    agg["far_ok"] = np.where(agg.n_deployed > 0, agg.n_met / agg.n_deployed.where(agg.n_deployed > 0), np.nan)   # v08: share of deployed designs
    agg["far_over_alpha"] = agg.far_test / agg.alpha; agg["source"] = src
    agg.to_csv(f"{OUT}/T_fresh_system.csv", index=False)
    st = []
    for (ds, a, cal), g in sel.groupby(["dataset", "alpha", "calib"]):
        g = g[g.p_deploy > 0]                       # paired over the splits in which both designs can be deployed
        mv = g[g.method == MAIN].set_index("seed"); 
        for m in PRIMARY:
            o = g[g.method == m].set_index("seed"); c = mv.index.intersection(o.index)
            if len(c) < 5: st.append(dict(dataset=ds, alpha=a, calib=cal, method=m, n=len(c))); continue
            w = (mv.p_deploy[c] * o.p_deploy[c]).values                       # v12: probability that both are deployed
            d = (mv.frr_test[c] - o.frr_test[c]).values; mean, ci, p = corrected_t_w(d, w)
            ow = float((w * o.frr_test[c].values).sum() / w.sum())
            st.append(dict(dataset=ds, alpha=a, calib=cal, method=m, n=len(c), n_eff=float(w.sum() ** 2 / (w ** 2).sum()),
                           mean_diff=mean, ci_lo=ci[0], ci_hi=ci[1], rel_diff=mean / ow if ow > 0 else np.nan, p_corr_t=p,
                           n_better=int((d < 0).sum()), n_worse=int((d > 0).sum())))
    st = pd.DataFrame(st); st["p_holm"] = np.nan
    for col in ["mean_diff", "ci_lo", "ci_hi", "rel_diff", "p_corr_t", "n_better", "n_worse"]:
        if col not in st: st[col] = np.nan
    for _, g in st[st.p_corr_t.notna()].groupby(["dataset", "alpha", "calib"]):
        st.loc[g.index, "p_holm"] = holm(g.p_corr_t.values)
    st["source"] = src; st.to_csv(f"{OUT}/T_fresh_stats.csv", index=False)
    # compliance of the deployed (selected) proposed designs: original ten splits (E3b) + fresh splits
    old = pd.read_csv(f"{OUT}/T_rev_selected.csv"); old = old[old.method == MAIN].assign(calib="boot", split_set="original 0-9")
    old = old.assign(p_deploy=1.0)
    new = sel[sel.method == MAIN].assign(split_set=lambda x: np.where(x.dataset == "lfw_x_fing", "D4 0-9", "fresh"))
    cols = ["dataset", "seed", "alpha", "calib", "p_deploy", "far_ok", "far_test", "frr_test", "split_set"]
    comp = pd.concat([old[cols], new[cols]], ignore_index=True)
    comp["met"] = comp.p_deploy * comp.far_ok.fillna(0)          # deployed and test FAR <= alpha (tie-averaged)
    comp["pfar"] = comp.p_deploy * comp.far_test.fillna(0); comp["pfrr"] = comp.p_deploy * comp.frr_test.fillna(0)
    cagg = comp.groupby(["dataset", "alpha", "calib", "split_set"]).agg(n=("seed", "size"), n_deployed=("p_deploy", "sum"), n_met=("met", "sum"),
                                                                        pfar=("pfar", "sum"), pfrr=("pfrr", "sum")).reset_index()
    cagg["far_test"] = cagg.pfar / cagg.n_deployed; cagg["frr_test"] = cagg.pfrr / cagg.n_deployed   # v12: deployment-conditional means
    cagg = cagg.drop(columns=["pfar", "pfrr"])
    cagg["far_ok"] = cagg.n_met / cagg.n_deployed                 # share of deployed designs with test FAR <= alpha
    cagg["far_over_alpha"] = cagg.far_test / cagg.alpha; cagg["source"] = f"T_rev_selected.csv + {src}"
    cagg.to_csv(f"{OUT}/T_fresh_compliance.csv", index=False)
    print(nseeds.to_string())
    print(agg[agg.calib == "boot"].pivot_table(index=["dataset", "alpha"], columns="method", values="frr_test").round(4).to_string())
    print(agg[agg.calib == "xfit"].pivot_table(index=["dataset", "alpha"], columns="method", values="frr_test").round(4).to_string())
    print(cagg.round(3).to_string())
    print(st[["dataset", "alpha", "calib", "method", "n", "mean_diff", "ci_lo", "ci_hi", "p_holm", "n_better", "n_worse"]].round(4).to_string())


if __name__ == "__main__":
    main()
