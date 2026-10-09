"""IJIS v09 (review M1, control arm): analysis of results/E3foldA (run_foldA.py) against the paper protocol (boot) and
the held-out calibration (xfit) of results/E3fresh.
(1) Reproduction: every fold-A design of run_foldA.py must equal the fold-A design of the xfit arm of run_fresh.py
    (same feasibility on fold A, same fold-A FAR and FRR, same thresholds before the final stage up to 1e-9 relative).
(2) Selection of the fold-A-only arm: per (dataset, split, alpha, method) the order with the smallest fold-A FRR among
    the designs feasible on fold A, ties averaged -- the same orders as in the xfit arm (taken from it where a fold-A rate
    was not reproduced) -- deployed with its fold-A
    calibration (fold B unused), so a design is deployed in every split in which one is feasible on fold A.
(3) Per arm (boot, foldA, xfit) and (dataset, alpha, method): splits, deployed designs, designs that met alpha on the
    test half, share of deployed designs that met it, mean test FRR and FAR.  Paired differences of the test FRR of
    the proposed design: foldA - boot over all splits, xfit - foldA over the splits in which xfit deployed a design.
    Compliance of the three arms on the splits in which xfit deployed every / none of the tied fold-A orders.
(4) Pooled over subsets and requirements, per method: share of splits with a compliant deployed design and share of the
    deployed designs that met alpha.
Outputs: results/tables/T_foldA_repro.csv, T_foldA_selected.csv, T_foldA_system.csv, T_foldA_paired.csv, T_foldA_methods.csv"""
import glob, json, numpy as np, pandas as pd
from analyze_rev import MAIN
R = "../results"; OUT = f"{R}/tables"
KEYS = ["dataset", "seed", "alpha", "method"]


def flat(s):
    out = []
    for t in json.loads(s):
        out += list(t) if isinstance(t, list) else [t]
    return np.array(out, float)


def same_pre(a, b):
    x, y = json.loads(a)[:-1], json.loads(b)[:-1]
    if len(x) != len(y): return False
    if not x: return True
    x, y = flat(json.dumps(x)), flat(json.dumps(y))
    fin = np.isfinite(x) & np.isfinite(y)
    return bool(np.array_equal(np.isfinite(x), np.isfinite(y)) and np.allclose(x[fin], y[fin], rtol=1e-9, atol=0))


def main():
    fs = sorted(glob.glob(f"{R}/E3foldA/foldA_*.csv")); A = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    src = f"results/E3foldA/foldA_*.csv ({len(fs)} files)"
    X = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/E3fresh/fresh_*.csv"))], ignore_index=True)
    par = X.kind == "parallel"; X.loc[par, "stages_gen"] = X.loc[par, "n_stages"].astype(float); X.loc[par, "stages_imp"] = X.loc[par, "n_stages"].astype(float)
    xf = X[X.calib == "xfit"]
    # (1) reproduction
    m = A.merge(xf, on=["dataset", "seed", "alpha", "order", "method"], how="outer", suffixes=("", "_x"), indicator=True)
    feasA = m.feasible.fillna(False).astype(bool); feasA_x = m.frr_foldA_x.notna()
    f = m[feasA & feasA_x]
    rates = np.isclose(f.frr_foldA, f.frr_foldA_x, rtol=0, atol=1e-12) & np.isclose(f.far_foldA, f.far_foldA_x, rtol=0, atol=1e-12)
    g = m[feasA & (m.feasible_x == True)]
    pre = np.array([same_pre(a, b) for a, b in zip(g.thr, g.thr_x)])
    rep = pd.DataFrame([dict(designs=len(m), both_files=int((m._merge == "both").sum()), feasibility_equal=int((feasA == feasA_x).sum()),
                             feasible_foldA=int(feasA.sum()), rates_equal=int(rates.sum()), pre_thresholds_compared=len(g),
                             pre_thresholds_equal=int(pre.sum()), source=src)])
    rep.to_csv(f"{OUT}/T_foldA_repro.csv", index=False)
    # (2) selection of the fold-A-only arm (same orders as xfit)
    rows = []
    for key, gg in A.groupby(KEYS):
        a = key[2]; ff = gg[gg.feasible == True]
        if len(ff) == 0: rows.append(dict(zip(KEYS, key), n_tied=0, p_deploy=0.0)); continue
        t = ff[np.isclose(ff.frr_foldA, ff.frr_foldA.min(), rtol=0, atol=1e-12)]
        rows.append(dict(zip(KEYS, key), n_tied=len(t), p_deploy=1.0, orders="|".join(t.order.astype(str)), frr_test=t.frr_test.mean(),
                         far_test=t.far_test.mean(), far_ok=float((t.far_test <= a).mean()), stages_gen=t.stages_gen.mean(), stages_imp=t.stages_imp.mean()))
    S = pd.DataFrame(rows).assign(calib="foldA")
    old = pd.read_csv(f"{OUT}/T_fresh_selected.csv").drop(columns="source")
    # the tied fold-A orders must be the same in both arms
    xo = old[old.calib == "xfit"].set_index(KEYS).orders; so = S.set_index(KEYS).orders; k = so.index.intersection(xo.index)
    eq = so.loc[k].fillna("") == xo.loc[k].fillna("")
    rep["selected_orders_equal"] = int(eq.sum()); rep["selected_compared"] = len(k)
    rep["selected_orders_differ"] = "; ".join(f"{i[0]} s{i[1]} {i[2]:g} {i[3]}: {so.loc[i]} vs {xo.loc[i]}" for i in eq.index[~eq])
    rep.to_csv(f"{OUT}/T_foldA_repro.csv", index=False)
    # the arm deploys exactly the orders fixed on fold A in the xfit arm (they differ from the rerun's own selection only
    # where a fold-A rate was not reproduced, see T_foldA_repro.csv); their fold-A calibrations come from the rerun
    Ai = A[A.feasible == True].set_index(["dataset", "seed", "alpha", "method", "order"])
    for i in eq.index[~eq]:
        r_ = S.index[(S.dataset == i[0]) & (S.seed == i[1]) & np.isclose(S.alpha, i[2]) & (S.method == i[3])][0]
        t = Ai.loc[[(i[0], i[1], i[2], i[3], o) for o in xo.loc[i].split("|")]]; a = i[2]
        S.loc[r_, ["n_tied", "orders", "frr_test", "far_test", "far_ok", "stages_gen", "stages_imp"]] = [
            len(t), xo.loc[i], t.frr_test.mean(), t.far_test.mean(), float((t.far_test <= a).mean()), t.stages_gen.mean(), t.stages_imp.mean()]
    cols = KEYS + ["calib", "p_deploy", "orders", "frr_test", "far_test", "far_ok", "stages_gen", "stages_imp"]
    SEL = pd.concat([old[cols], S[cols]], ignore_index=True)
    SEL["met"] = SEL.p_deploy * SEL.far_ok.fillna(0)
    SEL.assign(source=f"T_fresh_selected.csv + {src}").to_csv(f"{OUT}/T_foldA_selected.csv", index=False)
    # (3) system summary
    agg = SEL.groupby(["dataset", "alpha", "calib", "method"]).agg(n_splits=("seed", "size"), n_deployed=("p_deploy", "sum"), n_met=("met", "sum")).reset_index()
    dep = SEL[SEL.p_deploy > 0].groupby(["dataset", "alpha", "calib", "method"]).agg(frr_test=("frr_test", "mean"), far_test=("far_test", "mean")).reset_index()
    agg = agg.merge(dep, on=["dataset", "alpha", "calib", "method"], how="left")
    agg["far_ok"] = agg.n_met / agg.n_deployed.where(agg.n_deployed > 0); agg["met_share"] = agg.n_met / agg.n_splits
    agg["far_over_alpha"] = agg.far_test / agg.alpha; agg["source"] = f"T_fresh_selected.csv + {src}"
    agg.to_csv(f"{OUT}/T_foldA_system.csv", index=False)
    # paired FRR differences of the proposed design
    P = SEL[SEL.method == MAIN].pivot_table(index=["dataset", "alpha", "seed"], columns="calib", values="frr_test")
    D = SEL[SEL.method == MAIN].pivot_table(index=["dataset", "alpha", "seed"], columns="calib", values="p_deploy")
    pr = []
    K = SEL[SEL.method == MAIN].pivot_table(index=["dataset", "alpha", "seed"], columns="calib", values="met")
    for (ds, a), g in P.groupby(level=[0, 1]):
        d = D.loc[g.index]; k_ = K.loc[g.index]
        full = d.xfit >= 1 - 1e-12; none = d.xfit <= 1e-12                   # every / no tied fold-A order deployed by xfit
        pr.append(dict(dataset=ds, alpha=a, n=len(g), foldA_minus_boot=float((g.foldA - g.boot).mean()),
                       foldA_rel_boot=float((g.foldA - g.boot).mean() / g.boot.mean()),
                       n_xfit=int(full.sum()), xfit_minus_foldA=float((g.xfit - g.foldA)[full].mean()) if full.any() else np.nan,
                       xfit_rel_foldA=float((g.xfit - g.foldA)[full].mean() / g.foldA[full].mean()) if full.any() else np.nan,
                       met_boot_full=float(k_.boot[full].sum()), met_foldA_full=float(k_.foldA[full].sum()), met_xfit_full=float(k_.xfit[full].sum()),
                       n_none=int(none.sum()), met_foldA_none=float(k_.foldA[none].sum()), met_boot_none=float(k_.boot[none].sum())))
    pr = pd.DataFrame(pr); pr["source"] = f"T_foldA_selected.csv (proposed design); xfit pairs: splits in which every tied fold-A order was deployed"
    pr.to_csv(f"{OUT}/T_foldA_paired.csv", index=False)
    # pooled over subsets and requirements, per method (HYP excluded: rarely feasible)
    pm = agg[agg.method != "HYP"].groupby(["method", "calib"])[["n_splits", "n_deployed", "n_met"]].sum().reset_index()
    pm["met_share"] = pm.n_met / pm.n_splits; pm["far_ok"] = pm.n_met / pm.n_deployed; pm["source"] = f"T_foldA_system.csv"
    pm.to_csv(f"{OUT}/T_foldA_methods.csv", index=False)
    pd.set_option("display.width", 250)
    print(rep.drop(columns="source").to_string())
    print(agg[agg.method == MAIN].pivot_table(index=["dataset", "alpha"], columns="calib", values=["n_deployed", "n_met", "far_ok", "frr_test", "far_over_alpha"]).round(3).to_string())
    print(pr.drop(columns="source").round(4).to_string())
    print(agg.pivot_table(index=["dataset", "alpha", "method"], columns="calib", values="met_share").round(2).to_string())


if __name__ == "__main__":
    main()
