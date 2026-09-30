"""All tables for Stage 3 (CSV in results/tables).  Every table row records its source files."""
import glob, os, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from analyze import holm, rank_biserial

R = "../results"; OUT = f"{R}/tables"; os.makedirs(OUT, exist_ok=True)
MAIN = "LR-P1-N2"
def load(pat):
    fs = sorted(glob.glob(pat)); return (pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()), len(fs)

def chain_type(order):
    ms = order.split(">")
    if len(ms) == 1: return "single"
    face = sum(m.startswith("face") for m in ms); fing = len(ms) - face
    return "correlated-pair" if (face >= 2 or fing >= 2) else "cross-modal"

def select(df, crit_col):
    f = df[df.feasible == True].copy()
    f["crit"] = np.where(f.kind == "GP", f[crit_col], f.frr_train)
    return f.loc[f.groupby(["dataset", "seed", "alpha", "method"])["crit"].idxmin()]

def system_table(df, crit_col, name, src):
    sel = select(df, crit_col)
    feas = df.groupby(["dataset", "alpha", "method"])["feasible"].mean().rename("feas_rate")
    agg = sel.groupby(["dataset", "alpha", "method"]).agg(
        frr_test=("frr_test", "mean"), frr_test_sd=("frr_test", "std"), far_test=("far_test", "mean"),
        far_test_sd=("far_test", "std"), far_ok_test=("far_test", lambda s: np.nan),
        frr_pred=(crit_col, "mean"), stages_gen=("stages_gen", "mean"), stages_imp=("stages_imp", "mean"),
        n_seeds=("seed", "nunique")).join(feas).reset_index()
    ok = sel.assign(ok=sel.far_test <= sel.alpha * (1 + 1e-12)).groupby(["dataset", "alpha", "method"])["ok"].mean()
    agg = agg.drop(columns="far_ok_test").merge(ok.rename("far_ok_test_sel").reset_index(), on=["dataset", "alpha", "method"])
    allok = df[df.feasible == True].assign(ok=lambda d: d.far_test <= d.alpha * (1 + 1e-12)).groupby(["dataset", "alpha", "method"])["ok"].mean()
    agg = agg.merge(allok.rename("far_ok_test_all").reset_index(), on=["dataset", "alpha", "method"])
    agg["source"] = src; agg.to_csv(f"{OUT}/{name}.csv", index=False)
    # statistics vs MAIN
    stats = []
    for (ds, al), grp in df.groupby(["dataset", "alpha"]):
        s_main = sel[(sel.dataset == ds) & (sel.alpha == al) & (sel.method == MAIN)].set_index("seed")["frr_test"]
        a_main = grp[(grp.method == MAIN) & (grp.feasible == True)].set_index(["seed", "order"])["frr_test"]
        for mth in sorted(grp.method.unique()):
            if mth == MAIN: continue
            s_o = sel[(sel.dataset == ds) & (sel.alpha == al) & (sel.method == mth)].set_index("seed")["frr_test"]
            c = s_main.index.intersection(s_o.index); p_sel = np.nan
            if len(c) >= 5 and np.any(s_main[c].values != s_o[c].values):
                p_sel = wilcoxon(s_main[c], s_o[c]).pvalue
            a_o = grp[(grp.method == mth) & (grp.feasible == True)].set_index(["seed", "order"])["frr_test"]
            ci = a_main.index.intersection(a_o.index); p_all = rb = md = np.nan
            if len(ci) >= 5 and np.any(a_main[ci].values != a_o[ci].values):
                p_all = wilcoxon(a_main[ci], a_o[ci]).pvalue; rb = rank_biserial(a_main[ci], a_o[ci])
            if len(ci): md = float(np.median(a_main[ci].values - a_o[ci].values))
            stats.append({"dataset": ds, "alpha": al, "method": mth, "n_sel": len(c),
                          "mean_diff_sel": float((s_main[c] - s_o[c]).mean()) if len(c) else np.nan, "p_sel": p_sel,
                          "n_all": len(ci), "median_diff_all": md, "p_all": p_all, "rank_biserial_all": rb})
    st = pd.DataFrame(stats)
    for col in ["p_sel", "p_all"]:
        st[col + "_holm"] = np.nan
        for _, g in st.groupby(["dataset", "alpha"]):
            m = g[col].notna()
            if m.any(): st.loc[g.index[m], col + "_holm"] = holm(g.loc[m, col].values)
    st["source"] = src; st.to_csv(f"{OUT}/{name}_stats.csv", index=False)
    return agg, st

def main():
    e1, n1 = load(f"{R}/E1/main_*.csv"); e3, n3 = load(f"{R}/E3/main_*.csv"); cal, nc = load(f"{R}/E3cal/main_*.csv")
    print("files:", n1, n3, nc)
    # T_fit
    # per (dataset, seed, matcher) ratios to the proposed method, then geometric mean
    base = e1[e1.method == MAIN].set_index(["dataset", "seed", "matcher"])[["sse_dom", "maxerr_dom"]]
    e1 = e1.join(base, on=["dataset", "seed", "matcher"], rsuffix="_main")
    e1["sse_ratio"] = e1.sse_dom / e1.sse_dom_main; e1["maxerr_ratio"] = e1.maxerr_dom / e1.maxerr_dom_main
    gm = lambda x: float(np.exp(np.mean(np.log(np.clip(x, 1e-300, None)))))
    t = e1.groupby(["dataset", "method"]).agg(sse_ratio_gm=("sse_ratio", gm), maxerr_ratio_gm=("maxerr_ratio", gm),
          sse_med=("sse_dom", "median"), maxerr_med=("maxerr_dom", "median"),
          logarea=("logarea", "mean"), logarea_sd=("logarea", "std"),
          test_viol=("test_viol_rate", "mean"), test_viol_sd=("test_viol_rate", "std"), fit_time=("total_time", "mean"),
          cg_rounds=("rounds", "mean"), dominates=("dominates_all", "mean"), gap=("gap", "max"), n=("seed", "nunique")).reset_index()
    t["source"] = "results/E1/main_*.csv"; t.to_csv(f"{OUT}/T_fit.csv", index=False)
    # system: plain (TDSC-style) and calibrated
    e3["feasible"] = e3.feasible.astype(bool)
    system_table(e3, "frr_pred_gp", "T_system_plain", "results/E3/main_*.csv")
    if len(cal):
        cal["feasible"] = cal.feasible.astype(bool)
        system_table(cal, "frr_pred_cal", "T_system_cal", "results/E3cal/main_*.csv")
    # conservativeness: plain vs calibrated, all feasible GP designs, by chain type
    rows = []
    for reg, df, pc in [("plain", e3, "frr_pred_gp"), ("calibrated", cal, "frr_pred_cal")]:
        if not len(df): continue
        g = df[(df.kind == "GP") & (df.feasible == True)].copy(); g["chain"] = g.order.map(chain_type)
        g["far_ok_train"] = g.far_train <= g.alpha * (1 + 1e-12); g["far_ok_test"] = g.far_test <= g.alpha * (1 + 1e-12)
        g["frr_ok_test"] = g.frr_test <= g[pc] + 1e-12; g["far_ratio"] = g.far_test / g.alpha
        g["pred_over_real"] = g[pc] / g.frr_test.clip(lower=1e-6)
        for keys, h in list(g.groupby(["dataset", "alpha", "chain"])) + [((ds, al, "all"), h2) for (ds, al), h2 in g.groupby(["dataset", "alpha"])]:
            for mth, hh in [("all GP", h), (MAIN, h[h.method == MAIN])]:
                if not len(hh): continue
                rows.append({"regime": reg, "dataset": keys[0], "alpha": keys[1], "chain": keys[2], "method": mth, "n": len(hh),
                             "far_ok_train": hh.far_ok_train.mean(), "far_ok_test": hh.far_ok_test.mean(),
                             "far_ratio_median": hh.far_ratio.median(), "far_ratio_p95": hh.far_ratio.quantile(0.95),
                             "frr_ok_test": hh.frr_ok_test.mean()})
    cons = pd.DataFrame(rows); cons["source"] = "results/E3/main_*.csv; results/E3cal/main_*.csv"
    cons.to_csv(f"{OUT}/T_conservativeness.csv", index=False)
    # decomposition (plain; LR-P1-N2): medians of the five quantities
    g = e3[(e3.kind == "GP") & (e3.feasible == True) & (e3.method == MAIN)].copy(); g["chain"] = g.order.map(chain_type)
    dec = g.groupby(["dataset", "alpha"])[["frr_pred_gp", "frr_pred_exact", "frr_emp_model", "frr_train", "frr_test"]].median().reset_index()
    dec["source"] = "results/E3/main_*.csv (method LR-P1-N2)"; dec.to_csv(f"{OUT}/T_decomposition.csv", index=False)
    # noise robustness (D1, plain): FAR/FRR on noisy test scores
    if "frr_test_n05" in e3:
        nz = e3[(e3.dataset == "fing_x_face") & (e3.feasible == True) & (e3.kind == "GP")]
        nt = nz.groupby(["alpha", "method"])[["far_test", "far_test_n05", "far_test_n10", "frr_test", "frr_test_n05", "frr_test_n10"]].mean().reset_index()
        nt["source"] = "results/E3/main_fing_x_face_*.csv"; nt.to_csv(f"{OUT}/T_noise.csv", index=False)
    # E2 solver comparison
    e2, n2 = load(f"{R}/E2/e2_*.csv")
    if len(e2):
        s = e2.groupby(["dataset", "obj", "method"]).agg(rel_to_exact_mean=("rel_to_exact", "mean"), rel_to_exact_max=("rel_to_exact", "max"),
              cert_gap_mean=("cert_gap", "mean"), cert_gap_max=("cert_gap", "max"), time_mean=("time", "mean"), time_sd=("time", "std"),
              lb_valid=("lb_valid", lambda x: np.nan if x.isna().all() else float(np.mean(x.dropna().astype(bool)))),
              nodes=("nodes", "mean"), n=("seed", "size")).reset_index()
        s["source"] = "results/E2/e2_*.csv"; s.to_csv(f"{OUT}/T_solver.csv", index=False)
    e7, n7 = load(f"{R}/E7/e7_*.csv")
    if len(e7):
        e7.to_csv(f"{OUT}/T_scalability_raw.csv", index=False)
        e7["complete"] = e7["complete"].astype(str).str.lower().eq("true")
        e7["setting"] = np.select([e7.which == "N", e7.which == "D"], [e7.N, e7.get("nbins", e7.nD)], e7.nB)
        key = ["dataset", "matcher", "which", "setting"]; rows = []
        for k, g in e7.groupby(key):
            lr = g[g.method == "LR-BB + OBBT"].iloc[0]; en = g[g.method == "ENUM"].iloc[0]; de = g[g.method == "DE"].iloc[0]
            en_t = en.time if en.complete else en.enum_time_extrapolated
            rows.append(dict(zip(key, k), N=lr.N, nD=lr.nD, nB=lr.nB, lr_time=lr.time, lr_complete=lr.complete, lr_cert_gap=lr.cert_gap,
                             enum_time=en_t, enum_complete=en.complete, enum_fraction=en.get("enum_fraction", np.nan),
                             speedup=en_t / lr.time if lr.complete else np.nan, de_time=de.time,
                             grid_vs_de=(lr.value - de.value) / de.value if lr.complete else np.nan))
        t7 = pd.DataFrame(rows); t7["source"] = "results/E7/e7_*.csv (enum_time = extrapolated when enum_complete is False)"
        t7.to_csv(f"{OUT}/T_scalability.csv", index=False)
    # ablations / sensitivity / fraction (plain regime + fit metrics)
    ab = []
    for tag in ["abl_sample", "abl_region", "abl_N", "sens_b001", "sens_b005", "sens_nb60", "sens_nb240", "frac03", "frac07"]:
        d3, k3 = load(f"{R}/E3/{tag}_*.csv"); d1, k1 = load(f"{R}/E1/{tag}_*.csv")
        if not len(d3): continue
        d3["feasible"] = d3.feasible.astype(bool)
        sel = select(d3, "frr_pred_gp")
        a = sel.groupby(["alpha", "method"]).agg(frr_test=("frr_test", "mean"), frr_test_sd=("frr_test", "std"),
                                                  far_test=("far_test", "mean"), n=("seed", "nunique")).reset_index()
        f = d3[(d3.kind == "GP") & d3.feasible]
        a = a.merge(f.assign(ok=f.frr_train <= f.frr_pred_gp + 1e-12).groupby(["alpha", "method"])["ok"].mean().rename("frr_ok_train").reset_index(), on=["alpha", "method"])
        a = a.merge(f.assign(ok=f.frr_test <= f.frr_pred_gp + 1e-12).groupby(["alpha", "method"])["ok"].mean().rename("frr_ok_test").reset_index(), on=["alpha", "method"])
        a = a.merge(d3.groupby(["alpha", "method"])["feasible"].mean().rename("feas").reset_index(), on=["alpha", "method"])
        if len(d1):
            fm = d1.groupby("method").agg(logarea=("logarea", "mean"), test_viol=("test_viol_rate", "mean"),
                                          fit_time=("total_time", "mean"), gap_max=("gap", "max")).reset_index()
            a = a.merge(fm, on="method", how="left")
        a.insert(0, "tag", tag); a["source"] = f"results/E3/{tag}_*.csv; results/E1/{tag}_*.csv"; ab.append(a)
    if ab: pd.concat(ab).to_csv(f"{OUT}/T_ablation_sensitivity.csv", index=False)
    sp, ns = load(f"{R}/S3/sprt_*.csv")
    if len(sp):
        sp["feasible"] = sp.feasible.astype(bool); sp["kind"] = "rule"
        s = select(sp, "frr_train").groupby("alpha").agg(frr_test=("frr_test", "mean"), frr_test_sd=("frr_test", "std"),
                                                         far_test=("far_test", "mean"), n=("seed", "nunique")).reset_index()
        s["source"] = "results/S3/sprt_*.csv"; s.to_csv(f"{OUT}/T_sprt.csv", index=False)
    print("tables written")

if __name__ == "__main__":
    main()
