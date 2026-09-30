"""IJIS revision (mock reviewer 1, comment 3c): presentation-attack exposure per TRAIT instead of per matcher.
Face matchers C and G compare the same face image, so a face artifact replaces BOTH face scores; a fingerprint
artifact replaces the score of that finger.  A perfect spoof takes the scores of one randomly drawn genuine test
comparison (the same comparison for all matchers of the trait, so their correlation is kept); the other traits keep
the impostor's own scores.
Part A: designs selected in E3b at alpha = 1e-3 (ties averaged), all eight methods -> results/tables/T_rev_spoof_trait.csv
Part B: stage-capped GP and SPRT designs of E8 (thresholds stored in results/E8)   -> results/tables/T_rev_cost_spoof_trait.csv
Usage: python spoof_trait.py [dataset ...]   (default: all three)"""
import json, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from experiment_core import make_blocks
from serial_design import simulate, llr_function
from data import MATCHERS, data_file

TRAITS = {"fing_x_face": {"face": ["face_C", "face_G"], "right index": ["ri_V"], "left index": ["li_V"]},
          "fing_x_fing": {"right index": ["ri_V"], "left index": ["li_V"]},
          "face_x_face": {"face": ["face_C", "face_G"]},
          "lfw_x_fing": {"face": ["face_S"], "right index": ["ri_V"], "left index": ["li_V"]}}
METHODS = ["LR-P1-N2", "S1-Marcialis", "S2-Symmetric", "S3-SPRT", "S4-Direct", "P0-Parallel", "P1-LLR", "P2-LogReg"]
ALPHA = 1e-3


def serial_accept(sc, I, thr):
    fa, _, _, _ = simulate(sc, ~I, thr)   # simulate(scores, genuine mask, thr) returns the FAR over the non-genuine cells
    return fa


def run(ds):
    M = MATCHERS[ds]; D = dict(np.load(data_file(ds))); TR = TRAITS[ds]
    D4 = ds == "lfw_x_fing"                     # D4: designs of results/E3fresh (calib boot), selection T_fresh_selected
    sel = pd.read_csv("../results/tables/T_fresh_selected.csv" if D4 else "../results/tables/T_rev_selected.csv")
    if D4: sel = sel[sel.calib == "boot"]
    rowsA, rowsB = [], []
    for seed in range(10):
        e3 = pd.read_csv(f"../results/E3fresh/fresh_{ds}_s{seed}.csv") if D4 else pd.read_csv(f"../results/E3b/main_{ds}_s{seed}.csv")
        if D4: e3 = e3[e3.calib == "boot"]
        e3 = e3[(e3.feasible == True) & np.isclose(e3.alpha, ALPHA)]
        tr, te = make_blocks(D, M, seed); G, Gt = tr["_G"], te["_G"]; I = ~Gt
        rng = np.random.default_rng(2026 + seed)
        idx = rng.integers(0, int(Gt.sum()), size=Gt.shape)             # one genuine comparison per attempt, shared by the trait
        spoofed = {m: te[m][0][Gt][idx] for m in M}
        def scores(trait):                                              # test scores with `trait` spoofed
            return {m: (spoofed[m] if m in TR[trait] else te[m][0]) for m in M}
        llr = {m: llr_function(tr[m][0][G], tr[m][0][~G]) for m in M}
        # ---------------- Part A: selected designs
        for meth in METHODS:
            s = sel[(sel.dataset == ds) & (sel.seed == seed) & np.isclose(sel.alpha, ALPHA) & (sel.method == meth)]
            if len(s) == 0: continue
            rates = {t: [] for t in TR}
            for order in s.orders.iloc[0].split("|"):
                r = e3[(e3.method == meth) & (e3.order == order)].iloc[0]; thr = json.loads(r.thr)
                if meth.startswith("P"):
                    for t in TR:
                        sc = scores(t)
                        if meth == "P0-Parallel":
                            f = sum((sc[m] - tr[m][0][~G].mean()) / (tr[m][0][~G].std() + 1e-12) for m in M)
                        elif meth == "P1-LLR":
                            f = sum(llr[m](sc[m]) for m in M)
                        else:   # same class-balanced logistic regression as in spoof_rev.py / run_cal2.py
                            X_tr = np.stack([tr[m][0].ravel() for m in M], 1); mu_x, sd_x = X_tr.mean(0), X_tr.std(0) + 1e-12
                            y = G.ravel(); rg = np.random.default_rng(seed); imp_idx = np.nonzero(~y)[0]
                            sub = np.concatenate([np.nonzero(y)[0], rg.choice(imp_idx, min(len(imp_idx), 200000), replace=False)])
                            lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit((X_tr[sub] - mu_x) / sd_x, y[sub])
                            X = np.stack([sc[m].ravel() for m in M], 1); f = lr.decision_function((X - mu_x) / sd_x).reshape(Gt.shape)
                        rates[t].append(float((f[I] >= thr[0]).mean()))
                    continue
                chain = order.split(">"); thr = [tuple(x) if isinstance(x, list) else x for x in thr]
                for t in TR:
                    if not any(m in chain for m in TR[t]): rates[t].append(np.nan); continue
                    sc = scores(t); sc = [sc[m] for m in chain]
                    if meth == "S3-SPRT":
                        c = 0; cum = []
                        for m, x in zip(chain, sc): c = c + llr[m](x); cum.append(c)
                        sc = cum
                    rates[t].append(serial_accept(sc, I, thr))
            per = {t: (np.nanmean(v) if len(v) and not np.all(np.isnan(v)) else np.nan) for t, v in rates.items()}
            vals = [v for v in per.values() if np.isfinite(v)]
            rowsA.append({"dataset": ds, "seed": seed, "alpha": ALPHA, "method": meth, "spoof_trait_max": max(vals),
                          "spoof_trait_mean": float(np.mean(vals)), "n_traits": len(vals),
                          **{f"spoof_{t.replace(' ', '_')}": v for t, v in per.items()}})
        # ---------------- Part B: stage-capped designs of E8 (not run for D4)
        if D4: print(ds, seed, flush=True); continue
        e8 = pd.read_csv(f"../results/E8/cost_{ds}_s{seed}.csv")
        for _, r in e8[(e8.feasible == True) & e8.method.isin(["GP stage cap", "SPRT stage cap"])].iterrows():
            chain = r.order.split(">"); thr = [tuple(x) if isinstance(x, list) else x for x in json.loads(r.thr)]
            llr_c = [llr_function(tr[m][0][G], tr[m][0][~G]) for m in chain]
            worst = []
            for t in TR:
                if not any(m in chain for m in TR[t]): continue
                sc = scores(t); sc = [sc[m] for m in chain]
                if r.method == "SPRT stage cap":
                    c = 0; cum = []
                    for k, x in enumerate(sc): c = c + llr_c[k](x); cum.append(c)
                    sc = cum
                worst.append(serial_accept(sc, I, thr))
            rowsB.append({"dataset": ds, "seed": seed, "method": r.method, "param": r.param, "spoof_trait_worst": max(worst),
                          "spoof_matcher_worst_E8": r.spoof_worst})
        print(ds, seed, flush=True)
    return rowsA, rowsB


if __name__ == "__main__":
    dss = sys.argv[1:] or ["fing_x_face", "fing_x_fing", "face_x_face"]
    for ds in dss:
        a, b = run(ds)
        pd.DataFrame(a).to_csv(f"../results/tables/spoof_trait_raw_{ds}.csv", index=False)
        if b: pd.DataFrame(b).to_csv(f"../results/tables/cost_spoof_trait_raw_{ds}.csv", index=False)
    import glob
    A = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../results/tables/spoof_trait_raw_*.csv"))], ignore_index=True)
    agg = A.groupby(["dataset", "method"]).agg(spoof_trait_max=("spoof_trait_max", "mean"), spoof_trait_mean=("spoof_trait_mean", "mean"),
                                               n=("seed", "size")).reset_index()
    agg["source"] = np.where(agg.dataset == "lfw_x_fing", "results/E3fresh thresholds (calib boot) + test scores; spoof_trait.py (per-trait perfect spoof)",
                             "results/E3b thresholds + test scores; spoof_trait.py (per-trait perfect spoof)")
    agg.to_csv("../results/tables/T_rev_spoof_trait.csv", index=False)
    B = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../results/tables/cost_spoof_trait_raw_*.csv"))], ignore_index=True)
    B.to_csv("../results/tables/T_rev_cost_spoof_trait.csv", index=False)
    print(agg.round(3).to_string())
