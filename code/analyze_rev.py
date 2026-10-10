"""Revision analysis of E3b (subject-bootstrap calibration, extended baselines).
- Order selection for EVERY method: smallest training FRR of the deployed (calibrated) design; ties are
  averaged (expected result of a random tie-break).
- Inference over the ten overlapping subject splits: corrected resampled t-test (Nadeau & Bengio 2003) with
  95% confidence intervals of the paired mean difference; Holm correction within a pre-specified primary
  family (proposed design vs. each baseline), per subset and alpha.
Outputs results/tables/T_rev_*.csv (each with a `source` column)."""
import glob, numpy as np, pandas as pd
from scipy import stats

R = "../results"; OUT = f"{R}/tables"; MAIN = "LR-P1-N2"
PRIMARY = ["HYP", "S1-Marcialis", "S2-Symmetric", "S4-Direct", "S3-SPRT", "P0-Parallel", "P1-LLR", "P2-LogReg"]
SECONDARY = ["MONO", "MONO-cont", "DE-N2", "NLS-N2", "MAXMONO-K3", "LR-P1-N3", "LR-P2-N2", "LR-P1-N2-rel"]
N_TE_OVER_N_TR = 1.0          # 50/50 subject splits

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj

def corrected_t(d):
    """Nadeau-Bengio corrected resampled t-test for K paired differences from overlapping splits."""
    d = np.asarray(d, float); K = len(d); m = d.mean(); v = d.var(ddof=1)
    se = np.sqrt((1.0 / K + N_TE_OVER_N_TR) * v)
    if se == 0: return m, (m, m), (0.0 if m != 0 else 1.0)
    t = m / se; p = 2 * stats.t.sf(abs(t), K - 1); h = stats.t.ppf(0.975, K - 1) * se
    return m, (m - h, m + h), p

def corrected_t_w(d, w):
    """IJIS v12: corrected resampled t-test for paired differences d_j with weights w_j in (0, 1] (the probability that
    both designs are deployed in split j when each method breaks fold-A ties independently and at random before fold B
    is used).  Weighted mean; reliability-weighted variance; effective number of splits K_eff = (sum w)^2 / sum w^2 in
    place of K, with K_eff - 1 degrees of freedom.  With all weights equal to one it is corrected_t(d)."""
    d = np.asarray(d, float); w = np.asarray(w, float)
    if np.allclose(w, 1.0): return corrected_t(d)
    sw = w.sum(); m = float((w * d).sum() / sw); keff = sw ** 2 / (w ** 2).sum()
    v = float((w * (d - m) ** 2).sum() / (sw - (w ** 2).sum() / sw))
    se = np.sqrt((1.0 / keff + N_TE_OVER_N_TR) * v)
    if se == 0: return m, (m, m), (0.0 if m != 0 else 1.0)
    t = m / se; p = 2 * stats.t.sf(abs(t), keff - 1); h = stats.t.ppf(0.975, keff - 1) * se
    return m, (m - h, m + h), p

def load():
    fs = sorted(glob.glob(f"{R}/E3b/main_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True), fs

def select(df):
    """Tie-averaged selection per (dataset, seed, alpha, method)."""
    f = df[df.feasible == True].copy(); rows = []
    for key, g in f.groupby(["dataset", "seed", "alpha", "method"]):
        mn = g.frr_train.min(); t = g[np.isclose(g.frr_train, mn, rtol=0, atol=1e-12)]
        a = key[2]
        rows.append(dict(zip(["dataset", "seed", "alpha", "method"], key), n_tied=len(t), frr_test=t.frr_test.mean(),
                         far_test=t.far_test.mean(), far_ok=float((t.far_test <= a).mean()), frr_train=mn,
                         far_train=t.far_train.mean(), stages_gen=t.stages_gen.mean(), stages_imp=t.stages_imp.mean(),
                         frr_pred=t.frr_pred_cal.mean() if "frr_pred_cal" in t else np.nan,
                         pred_ok=float((t.frr_test <= t.frr_pred_cal).mean()) if t.frr_pred_cal.notna().all() else np.nan,
                         n_stages=t.n_stages.mean(), orders="|".join(t.order.astype(str))))
    return pd.DataFrame(rows)

def main():
    df, fs = load(); src = f"results/E3b/main_*.csv ({len(fs)} files)"
    sel = select(df); sel["source"] = src; sel.to_csv(f"{OUT}/T_rev_selected.csv", index=False)
    n_feas = df.groupby(["dataset", "alpha", "method"]).feasible.mean().rename("feas_rate")
    agg = sel.groupby(["dataset", "alpha", "method"]).agg(
        n_seeds=("seed", "size"), frr_test=("frr_test", "mean"), frr_test_sd=("frr_test", "std"),
        far_test=("far_test", "mean"), far_ok=("far_ok", "mean"), stages_gen=("stages_gen", "mean"),
        stages_imp=("stages_imp", "mean"), pred_ok=("pred_ok", "mean"), frr_pred=("frr_pred", "mean"),
        n_tied=("n_tied", "mean")).reset_index().merge(n_feas.reset_index(), on=["dataset", "alpha", "method"], how="left")
    agg["far_over_alpha"] = agg.far_test / agg.alpha; agg["source"] = src
    agg.to_csv(f"{OUT}/T_rev_system.csv", index=False)
    rows = []
    for (ds, a), g in sel.groupby(["dataset", "alpha"]):
        mainv = g[g.method == MAIN].set_index("seed").frr_test
        for fam, meths in [("primary", PRIMARY), ("secondary", SECONDARY)]:
            for m in meths:
                o = g[g.method == m].set_index("seed").frr_test; c = mainv.index.intersection(o.index)
                if len(c) < 5: rows.append(dict(dataset=ds, alpha=a, family=fam, method=m, n=len(c))); continue
                d = (mainv[c] - o[c]).values; mean, ci, p = corrected_t(d)
                w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                rows.append(dict(dataset=ds, alpha=a, family=fam, method=m, n=len(c), mean_diff=mean, ci_lo=ci[0], ci_hi=ci[1],
                                 rel_diff=mean / o[c].mean() if o[c].mean() > 0 else np.nan, p_corr_t=p, p_wilcoxon_naive=w,
                                 n_better=int((d < 0).sum()), n_worse=int((d > 0).sum())))
    st = pd.DataFrame(rows); st["p_holm"] = np.nan
    for col in ["mean_diff", "ci_lo", "ci_hi", "rel_diff", "p_corr_t", "p_wilcoxon_naive", "n_better", "n_worse"]:
        if col not in st: st[col] = np.nan
    for _, g in st[(st.family == "primary") & st.p_corr_t.notna()].groupby(["dataset", "alpha"]):
        st.loc[g.index, "p_holm"] = holm(g.p_corr_t.values)
    st["source"] = src; st.to_csv(f"{OUT}/T_rev_stats.csv", index=False)
    print(agg[agg.method.isin([MAIN] + PRIMARY)].pivot_table(index=["dataset", "alpha"], columns="method", values="frr_test").round(4).to_string())
    print(st[st.family == "primary"][["dataset", "alpha", "method", "mean_diff", "ci_lo", "ci_hi", "p_corr_t", "p_holm", "n_better", "n_worse"]].round(4).to_string())

if __name__ == "__main__":
    main()
