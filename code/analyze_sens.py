"""IJIS v08 (review M4, M5): analysis of results/E3sensall (written by calib_sensitivity_all.py).
- Check that the paper variant reproduces results/E3b for every primary design (feasibility, test FRR and FAR).
- Per variant: order selection by training FRR (ties averaged) for every method, as in analyze_rev.py; test FRR and
  FAR compliance of the selected designs; corrected resampled t-tests of the proposed design against the eight primary
  baselines with Holm correction per setting; decisions (significant lower / higher / none) compared with the paper.
- Monte Carlo stability: share of designs whose final threshold or feasibility changed relative to the paper variant,
  and the change of their training FAR in units of alpha.
- Clopper-Pearson floor of the GP designs: share of feasible designs in which it binds (all orders and selected
  designs, tie-weighted), and the 'nofloor' variant as an ablation with re-selection.
- Proposition 1 at every deployed stage threshold of the GP designs (paper variant, training data).
Outputs: results/tables/T_sens_system.csv, T_sens_stats.csv, T_sens_summary.csv, T_sens_lemma.csv"""
import glob, numpy as np, pandas as pd
from analyze_rev import corrected_t, holm, MAIN, PRIMARY
R = "../results"; OUT = f"{R}/tables"
VARS = ["paper", "B1000", "seed7000", "seed8000", "nofloor"]


def select(df):
    rows = []
    for key, g in df[df.feasible == True].groupby(["variant", "dataset", "seed", "alpha", "method"]):
        t = g[np.isclose(g.frr_train, g.frr_train.min(), rtol=0, atol=1e-12)]; a = key[3]
        rows.append(dict(zip(["variant", "dataset", "seed", "alpha", "method"], key), n_tied=len(t), frr_test=t.frr_test.mean(),
                         far_test=t.far_test.mean(), far_ok=float((t.far_test <= a).mean()), cp_binds=t.cp_binds.astype(float).mean()))
    return pd.DataFrame(rows)


def stats(sel):
    st = []
    for (v, ds, a), g in sel.groupby(["variant", "dataset", "alpha"]):
        mv = g[g.method == MAIN].set_index("seed").frr_test
        for m in PRIMARY:
            o = g[g.method == m].set_index("seed").frr_test; c = mv.index.intersection(o.index)
            if len(c) < 5: st.append(dict(variant=v, dataset=ds, alpha=a, method=m, n=len(c))); continue
            d = (mv[c] - o[c]).values; mean, ci, p = corrected_t(d)
            st.append(dict(variant=v, dataset=ds, alpha=a, method=m, n=len(c), mean_diff=mean, ci_lo=ci[0], ci_hi=ci[1], p_corr_t=p))
    st = pd.DataFrame(st); st["p_holm"] = np.nan
    for col in ["mean_diff", "ci_lo", "ci_hi", "p_corr_t"]:
        if col not in st: st[col] = np.nan
    for _, g in st[st.p_corr_t.notna()].groupby(["variant", "dataset", "alpha"]):
        st.loc[g.index, "p_holm"] = holm(g.p_corr_t.values)
    st["decision"] = np.where(st.p_holm < 0.05, np.where(st.mean_diff < 0, "lower", "higher"), np.where(st.p_holm.notna(), "none", "n/a"))
    return st


def main():
    fs = sorted(glob.glob(f"{R}/E3sensall/sensall_*.csv")); df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    src = f"results/E3sensall/sensall_*.csv ({len(fs)} files)"
    # nofloor: GP rows of the proposed design and HYP; every other method keeps its paper rows
    nf = df[df.variant == "paper"].copy(); nf = nf[nf.kind != "GP"]; nf["variant"] = "nofloor"
    df = pd.concat([df, nf], ignore_index=True)
    # (1) reproduction of E3b
    e3 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/E3b/main_*.csv"))], ignore_index=True)
    p = df[df.variant == "paper"].merge(e3[["dataset", "seed", "alpha", "order", "method", "feasible", "frr_test", "far_test"]],
                                        on=["dataset", "seed", "alpha", "order", "method"], how="left", suffixes=("", "_e3"))
    p["feasible_e3"] = p.feasible_e3.fillna(False).astype(bool)
    rep_feas = float((p.feasible.astype(bool) == p.feasible_e3).mean()); f = p[p.feasible == True]
    rep_frr = float((np.isclose(f.frr_test, f.frr_test_e3, rtol=0, atol=0) & np.isclose(f.far_test, f.far_test_e3, rtol=0, atol=0)).mean())
    # (2) selection, system, stats
    sel = select(df)
    sysr = sel.groupby(["variant", "dataset", "alpha", "method"]).agg(n=("seed", "size"), frr_test=("frr_test", "mean"),
                                                                      n_met=("far_ok", "sum"), cp_binds=("cp_binds", "mean")).reset_index()
    sysr["source"] = src; sysr.to_csv(f"{OUT}/T_sens_system.csv", index=False)
    st = stats(sel); st["source"] = src; st.to_csv(f"{OUT}/T_sens_stats.csv", index=False)
    ref = st[st.variant == "paper"].set_index(["dataset", "alpha", "method"])
    # published decisions (T_rev_stats) must equal the paper variant
    pub = pd.read_csv(f"{OUT}/T_rev_stats.csv"); pub = pub[(pub.family == "primary") & pub.p_holm.notna()].set_index(["dataset", "alpha", "method"])
    pub_dec = np.where(pub.p_holm < 0.05, np.where(pub.mean_diff < 0, "lower", "higher"), "none")
    ki = pub.index.intersection(ref.index)
    same_pub = float((pd.Series(pub_dec, index=pub.index).loc[ki] == ref.loc[ki, "decision"]).mean()) if len(ki) else np.nan
    summ = []
    for v in VARS:
        s_ = st[st.variant == v].set_index(["dataset", "alpha", "method"]); k = ref.index[ref.decision != "n/a"]
        kept = int((s_.loc[k, "decision"] == ref.loc[k, "decision"]).sum())
        # Monte Carlo stability against the paper variant (all designs, all orders)
        a_ = df[df.variant == v].set_index(["dataset", "seed", "alpha", "order", "method"])
        b_ = df[df.variant == "paper"].set_index(["dataset", "seed", "alpha", "order", "method"]); c = a_.index.intersection(b_.index)
        A, Bp = a_.loc[c], b_.loc[c]
        if v == "nofloor": gpm = (A.kind == "GP").values; A, Bp = A[gpm], Bp[gpm]       # the floor concerns the GP designs only
        both = (A.feasible == True) & (Bp.feasible == True)
        feas_changed = int((A.feasible.astype(bool) != Bp.feasible.astype(bool)).sum())
        thr_changed = float((~np.isclose(A.thr_final[both], Bp.thr_final[both], rtol=0, atol=0)).mean())
        al = np.asarray(A.index.get_level_values("alpha"), float)[both.values]
        dfar = pd.Series(np.abs(A.far_train.values[both.values] - Bp.far_train.values[both.values]) / al)
        selp = sel[(sel.variant == v) & (sel.method == MAIN)]
        summ.append(dict(variant=v, designs=len(A), feasible=int((A.feasible == True).sum()), feasibility_changed=feas_changed,
                         thr_changed_share=thr_changed, dfar_train_mean_alpha=float(dfar.mean()), dfar_train_max_alpha=float(dfar.max()),
                         decisions_compared=len(k), decisions_kept=kept, proposed_n_met=float(selp.far_ok.sum()), proposed_n=len(selp),
                         proposed_frr_mean=float(selp.frr_test.mean())))
    S = pd.DataFrame(summ)
    # (3) Clopper-Pearson floor (paper variant)
    gp = df[(df.variant == "paper") & (df.kind == "GP") & (df.feasible == True)]
    selp = sel[(sel.variant == "paper") & (sel.method == MAIN)]
    S["cp_binds_all_gp"] = float(gp[gp.method == MAIN].cp_binds.astype(float).mean())
    S["cp_binds_selected"] = float(selp.cp_binds.mean())
    S["reproduces_E3b_feasibility"] = rep_feas; S["reproduces_E3b_test"] = rep_frr; S["paper_equals_published_decisions"] = same_pub
    S["source"] = src; S.to_csv(f"{OUT}/T_sens_summary.csv", index=False)
    # (4) Proposition 1
    lf = sorted(glob.glob(f"{R}/E3sensall/lemma_*.csv")); L = pd.concat([pd.read_csv(x) for x in lf], ignore_index=True)
    LT = pd.DataFrame([dict(n=len(L), holds=int(L.holds.sum()), zero_far=int((L.far == 0).sum()), zero_far_holds=int(L[L.far == 0].holds.sum()),
                            max_ratio=float((L.frr / L.g).max()), source=f"results/E3sensall/lemma_*.csv ({len(lf)} files)")])
    LT.to_csv(f"{OUT}/T_sens_lemma.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.drop(columns="source").round(4).to_string()); print(LT.to_string())
    piv = sysr[sysr.method.isin([MAIN, "S3-SPRT", "P1-LLR"])].pivot_table(index=["dataset", "alpha", "method"], columns="variant", values=["frr_test", "n_met"])
    print(piv.round(4).to_string())
    ch = st.merge(st[st.variant == "paper"][["dataset", "alpha", "method", "decision"]], on=["dataset", "alpha", "method"], suffixes=("", "_paper"))
    print(ch[(ch.decision != ch.decision_paper)][["variant", "dataset", "alpha", "method", "n", "mean_diff", "p_holm", "decision", "decision_paper"]].to_string())
    print(sel[(sel.method == MAIN)].groupby(["variant", "dataset", "alpha"]).cp_binds.mean().unstack(0).round(3).to_string())


if __name__ == "__main__":
    main()
