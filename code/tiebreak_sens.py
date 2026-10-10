"""IJIS v12 (review of v11, R1): sensitivity of the held-out (xfit) results to the tie-break among the best fold-A orders.
The paper reports expectations over a uniformly random tie-break made before fold B is used (analyze_fresh.py).  Here
the tie is instead broken by a single random draw per method and split, as a fixed rule would do, for 1000 independent
draws (numpy default_rng(9000 + draw)).  For every draw: the proposed design's deployment count, compliance count and
mean test FRR and FAR / alpha of the deployed designs per subset and requirement; and the Holm decisions of the primary
comparisons (corrected resampled t over the splits in which both designs were deployed; a comparison with fewer than five
such splits is not tested in that draw), compared with those of the expectation-based analysis (T_fresh_stats.csv, calib xfit).
Outputs: results/tables/T_tiebreak_summary.csv, T_tiebreak_decisions.csv"""
import glob, numpy as np, pandas as pd
from analyze_rev import corrected_t, holm, MAIN
from analyze_fresh import PRIMARY
R = "../results"; OUT = f"{R}/tables"; NDRAW = 1000


def main():
    X = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/E3fresh/fresh_*.csv"))], ignore_index=True)
    X = X[X.calib == "xfit"].set_index(["dataset", "seed", "alpha", "method", "order"])
    S = pd.read_csv(f"{OUT}/T_fresh_selected.csv"); S = S[(S.calib == "xfit") & (S.design_foldA == True)]
    S = S[S.method.isin([MAIN] + PRIMARY)].reset_index(drop=True)
    # per setting: arrays over the tied fold-A orders (deployed flag, test FRR, test FAR)
    cand = []
    for _, r in S.iterrows():
        rows = X.loc[[(r.dataset, r.seed, r.alpha, r.method, o) for o in str(r.orders).split("|")]]
        cand.append((rows.feasible.fillna(False).astype(bool).values, rows.frr_test.values, rows.far_test.values))
    ref = pd.read_csv(f"{OUT}/T_fresh_stats.csv"); ref = ref[(ref.calib == "xfit") & ref.p_holm.notna()]
    ref["decision"] = np.where(ref.p_holm < 0.05, np.where(ref.mean_diff < 0, "lower", "higher"), "none")
    ref = ref.set_index(["dataset", "alpha", "method"]).decision
    summ, decs = [], []
    for b in range(NDRAW):
        rng = np.random.default_rng(9000 + b)
        pick = [rng.integers(len(c[0])) for c in cand]
        D = S[["dataset", "seed", "alpha", "method"]].copy()
        D["dep"] = [c[0][k] for c, k in zip(cand, pick)]
        D["frr"] = [c[1][k] if c[0][k] else np.nan for c, k in zip(cand, pick)]
        D["far"] = [c[2][k] if c[0][k] else np.nan for c, k in zip(cand, pick)]
        P = D[D.method == MAIN]
        for (ds, a), g in P.groupby(["dataset", "alpha"]):
            dd = g[g.dep]
            summ.append(dict(draw=b, dataset=ds, alpha=a, n_splits=len(g), deployed=int(g.dep.sum()), met=int((dd.far <= a).sum()),
                             frr_test=dd.frr.mean(), far_over_alpha=(dd.far / a).mean()))
        for (ds, a), g in D[D.dep].groupby(["dataset", "alpha"]):
            mv = g[g.method == MAIN].set_index("seed").frr; ps, ks = [], []
            for m in PRIMARY:
                o = g[g.method == m].set_index("seed").frr; c = mv.index.intersection(o.index)
                if len(c) < 5: continue
                mean, _, p = corrected_t((mv[c] - o[c]).values); ps.append(p); ks.append((m, mean))
            if not ps: continue
            ph = holm(np.array(ps))
            for (m, mean), p_ in zip(ks, ph):
                dec = "none" if p_ >= 0.05 else ("lower" if mean < 0 else "higher")
                key = (ds, a, m)
                decs.append(dict(draw=b, dataset=ds, alpha=a, method=m, decision=dec, reference=ref.get(key, "n/a")))
    Sm = pd.DataFrame(summ); Dc = pd.DataFrame(decs)
    Dc["agree"] = Dc.decision == Dc.reference
    agg = Sm.groupby(["dataset", "alpha"]).agg(n_splits=("n_splits", "first"), deployed_min=("deployed", "min"), deployed_max=("deployed", "max"),
                                               deployed_mean=("deployed", "mean"), met_min=("met", "min"), met_max=("met", "max"),
                                               frr_q05=("frr_test", lambda v: v.quantile(0.05)), frr_q95=("frr_test", lambda v: v.quantile(0.95)),
                                               far_alpha_q05=("far_over_alpha", lambda v: v.quantile(0.05)),
                                               far_alpha_q95=("far_over_alpha", lambda v: v.quantile(0.95))).reset_index()
    per_draw = Dc.groupby("draw").agree.agg(["sum", "size"])
    agg["draws"] = NDRAW; agg["source"] = "results/E3fresh xfit rows + T_fresh_selected.csv; single random tie-break per method and split"
    agg.to_csv(f"{OUT}/T_tiebreak_summary.csv", index=False)
    dk = Dc.groupby(["dataset", "alpha", "method", "reference"]).agg(agree_share=("agree", "mean"), n=("agree", "size"),
                                                                     share_lower=("decision", lambda v: (v == "lower").mean()),
                                                                     share_none=("decision", lambda v: (v == "none").mean()),
                                                                     share_higher=("decision", lambda v: (v == "higher").mean())).reset_index()
    dk["tested_share"] = dk.n / NDRAW            # share of draws with at least five splits in which both designs were deployed
    dk["decisions_per_draw_min"] = int(per_draw["size"].min()); dk["decisions_per_draw_max"] = int(per_draw["size"].max())
    dk["all_agree_share"] = float((per_draw["sum"] == per_draw["size"]).mean())      # every decision made in the draw agrees
    dk["disagree_max_per_draw"] = int((per_draw["size"] - per_draw["sum"]).max())
    dk["source"] = "tiebreak_sens.py, 1000 draws"; dk.to_csv(f"{OUT}/T_tiebreak_decisions.csv", index=False)
    pd.set_option("display.width", 250)
    print(agg.drop(columns="source").round(4).to_string())
    print(dk[dk.agree_share < 1].drop(columns="source").round(3).to_string())
    print("decisions per draw:", per_draw["size"].min(), "-", per_draw["size"].max(), "all agree in share of draws:", (per_draw["sum"] == per_draw["size"]).mean(),
          "max disagreements per draw:", (per_draw["size"] - per_draw["sum"]).max())


if __name__ == "__main__":
    main()
