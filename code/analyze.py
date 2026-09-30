"""Aggregate E1/E3 results: tables (CSV + LaTeX-ready), statistics, traceability entries."""
import glob, json, sys, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon, beta as betadist

RES = "../results"; OUT = "../results/tables"; os.makedirs(OUT, exist_ok=True)
MAIN = "LR-P1-N2"
GP_ORDER = ["HYP", "MONO", "MONO-rel", "NLS-N2", "DE-N2", "MAXMONO-K3", "LR-P1-N2", "LR-P1-N2-rel", "LR-P2-N2", "LR-P2-N2-rel", "LR-P1-N3"]
RULE_ORDER = ["S1-Marcialis", "S2-Symmetric", "S4-Direct", "P0-Parallel"]

def load(exp, tag="main"):
    fs = sorted(glob.glob(f"{RES}/{exp}/{tag}_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj

def rank_biserial(x, y):
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    if len(d) == 0: return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())

def cp_interval(k, n, conf=0.95):
    a = 1 - conf
    lo = betadist.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = betadist.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

def select_designs(e3):
    """Per (dataset, seed, alpha, method): the order chosen using TRAINING information only
    (GP: minimum predicted FRR; rules: minimum training FRR)."""
    f = e3[e3.feasible == True].copy()
    f["crit"] = np.where(f.kind == "GP", f.frr_pred_gp, f.frr_train)
    idx = f.groupby(["dataset", "seed", "alpha", "method"])["crit"].idxmin()
    return f.loc[idx]

def main():
    e1, e3 = load("E1"), load("E3")
    e3["feasible"] = e3["feasible"].astype(bool)
    # ---------------- T-fit: envelope tightness and generalisation
    t = e1.groupby(["dataset", "method"]).agg(logarea=("logarea", "mean"), logarea_sd=("logarea", "std"),
                                               test_viol=("test_viol_rate", "mean"), fit_time=("total_time", "mean"),
                                               dominates=("dominates_all", "mean"), n=("seed", "nunique")).reset_index()
    t.to_csv(f"{OUT}/T_fit.csv", index=False)
    # ---------------- T-system: selected designs
    sel = select_designs(e3)
    feas = e3.groupby(["dataset", "alpha", "method"])["feasible"].mean().rename("feas_rate")
    agg = sel.groupby(["dataset", "alpha", "method"]).agg(frr_test=("frr_test", "mean"), frr_test_sd=("frr_test", "std"),
                                                            far_test=("far_test", "mean"), frr_pred=("frr_pred_gp", "mean"),
                                                            stages_gen=("stages_gen", "mean"), stages_imp=("stages_imp", "mean"),
                                                            n_seeds=("seed", "nunique"))
    agg = agg.join(feas).reset_index()
    # Wilcoxon on selected designs (paired by seed) and on all commonly-feasible (seed, order) pairs
    stats = []
    for (ds, al), grp in e3.groupby(["dataset", "alpha"]):
        s_main = sel[(sel.dataset == ds) & (sel.alpha == al) & (sel.method == MAIN)].set_index("seed")["frr_test"]
        a_main = grp[(grp.method == MAIN) & grp.feasible].set_index(["seed", "order"])["frr_test"]
        for mth in grp.method.unique():
            if mth == MAIN: continue
            s_o = sel[(sel.dataset == ds) & (sel.alpha == al) & (sel.method == mth)].set_index("seed")["frr_test"]
            common = s_main.index.intersection(s_o.index)
            p_sel = np.nan
            if len(common) >= 5 and np.any(s_main[common].values != s_o[common].values):
                p_sel = wilcoxon(s_main[common], s_o[common], zero_method="wilcox").pvalue
            a_o = grp[(grp.method == mth) & grp.feasible].set_index(["seed", "order"])["frr_test"]
            ci = a_main.index.intersection(a_o.index); p_all = np.nan; rb = np.nan
            if len(ci) >= 5 and np.any(a_main[ci].values != a_o[ci].values):
                p_all = wilcoxon(a_main[ci], a_o[ci], zero_method="wilcox").pvalue; rb = rank_biserial(a_main[ci], a_o[ci])
            stats.append({"dataset": ds, "alpha": al, "method": mth, "n_sel": len(common), "p_sel": p_sel,
                          "n_all": len(ci), "p_all": p_all, "rank_biserial_all": rb,
                          "median_diff_all": float(np.median(a_main[ci].values - a_o[ci].values)) if len(ci) else np.nan})
    st = pd.DataFrame(stats)
    for col in ["p_sel", "p_all"]:
        st[col + "_holm"] = np.nan
        for (ds, al), g in st.groupby(["dataset", "alpha"]):
            m = g[col].notna()
            if m.any(): st.loc[g.index[m], col + "_holm"] = holm(g.loc[m, col].values)
    st.to_csv(f"{OUT}/T_system_stats.csv", index=False)
    agg.to_csv(f"{OUT}/T_system.csv", index=False)
    # ---------------- conservativeness (all feasible GP designs)
    gp = e3[(e3.kind == "GP") & e3.feasible].copy()
    gp["far_ok_test"] = gp.far_test <= gp.alpha * (1 + 1e-9)
    gp["far_ok_train"] = gp.far_train <= gp.alpha * (1 + 1e-9)
    gp["frr_ok_test"] = gp.frr_test <= gp.frr_pred_gp + 1e-12
    gp["frr_ok_train"] = gp.frr_train <= gp.frr_pred_gp + 1e-12
    cons = gp.groupby(["dataset", "alpha", "method"]).agg(n=("seed", "size"), far_ok_train=("far_ok_train", "mean"),
                                                          far_ok_test=("far_ok_test", "mean"), frr_ok_train=("frr_ok_train", "mean"),
                                                          frr_ok_test=("frr_ok_test", "mean"),
                                                          far_ratio_med=("far_test", lambda s: np.nan)).reset_index()
    ratio = gp.assign(r=gp.far_test / gp.alpha).groupby(["dataset", "alpha", "method"])["r"].agg(["median", "max"]).reset_index()
    cons = cons.drop(columns="far_ratio_med").merge(ratio, on=["dataset", "alpha", "method"])
    cons.to_csv(f"{OUT}/T_conservativeness.csv", index=False)
    dec = gp.groupby(["dataset", "alpha", "method"])[["frr_pred_gp", "frr_pred_exact", "frr_emp_model", "frr_train", "frr_test"]].median().reset_index()
    dec.to_csv(f"{OUT}/T_decomposition.csv", index=False)
    print(agg.round(4).to_string()); print(st.round(4).to_string()); print(cons.round(3).to_string())

if __name__ == "__main__":
    main()
