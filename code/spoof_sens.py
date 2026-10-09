"""IJIS v09 (review M8): sensitivity of the single-trait presentation-attack stress test (spoof_trait.py, Table 8) to
the strength of the artifact, to attacks that shift the scores above the genuine distribution, and to presentation-
attack detection (PAD).  Same designs, splits and spoofed comparisons as spoof_trait.py Part A: the designs selected
at alpha = 1e-3 (ties averaged; E3b for D1-D3, E3fresh 'boot' for D4), all eight methods, ten splits, one genuine test
comparison drawn per attempt and shared by the matchers of the spoofed trait (rng seed 2026 + split).
Attack models for a spoofed trait t (other traits keep the impostor's own scores):
  strength lam in {0, 0.25, 0.5, 0.75, 1}: s = (1 - lam) * s_impostor + lam * s_genuine   (lam = 1 is Table 8,
           lam = 0 the zero-effort FAR)
  shift    d in {0.5, 1}: s = s_genuine + d * sd_g, sd_g the standard deviation of the matcher's genuine TRAINING
           scores (an attack biased above the genuine distribution)
  PAD      perfect artifact (lam = 1) and a detector applied once per trait acquisition, independent of the scores:
           the artifact passes with probability APCER in {0.2, 0.05}; each bona fide presentation (genuine users, and
           the impostor's own traits) is rejected with probability BPCER = 0.01.  A trait is acquired at the first
           stage that uses one of its matchers (the two face matchers of D1/D3 compare one face image); parallel fusion
           acquires every trait.  An attack is accepted iff the chain accepts it and every PAD it meets passes.
For every model the acceptance rate is computed per trait and split; 'worst' is the largest over the traits the method
acquires (as in Table 8), then averaged over splits.  With PAD, the genuine test FRR including PAD rejections and the
mean number of trait acquisitions per genuine and per impostor claim are reported as well.
Output: results/tables/spoof_sens_raw_<dataset>.csv, results/tables/T_spoof_sens.csv"""
import json, sys, glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from experiment_core import make_blocks
from serial_design import llr_function
from data import MATCHERS, data_file
from spoof_trait import TRAITS, METHODS, ALPHA

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]; SHIFTS = [0.5, 1.0]; APCERS = [0.2, 0.05]; BPCER = 0.01


def paths(scores, thr):
    """Serial decision per cell: accepted (bool) and number of stages used (int)."""
    S = len(scores); shp = scores[0].shape
    decided = np.zeros(shp, bool); acc = np.zeros(shp, bool); used = np.zeros(shp, np.int8)
    for s in range(S):
        und = ~decided; used[und] += 1
        if s < S - 1:
            ta, tr = thr[s]; a = und & (scores[s] >= ta); r = und & (scores[s] < tr); acc |= a; decided |= a | r
        else:
            acc |= und & (scores[s] >= thr[s]); decided[:] = True
    return acc, used


def first_use(chain, mats):
    """1-based stage at which a trait (set of matchers) is first acquired in `chain`, or a large number."""
    k = [i + 1 for i, m in enumerate(chain) if m in mats]
    return min(k) if k else 10 ** 6


def run(ds):
    M = MATCHERS[ds]; D = dict(np.load(data_file(ds))); TR = TRAITS[ds]
    D4 = ds == "lfw_x_fing"
    sel = pd.read_csv("../results/tables/T_fresh_selected.csv" if D4 else "../results/tables/T_rev_selected.csv")
    if D4: sel = sel[sel.calib == "boot"]
    rows = []
    for seed in range(10):
        e3 = pd.read_csv(f"../results/E3fresh/fresh_{ds}_s{seed}.csv") if D4 else pd.read_csv(f"../results/E3b/main_{ds}_s{seed}.csv")
        if D4: e3 = e3[e3.calib == "boot"]
        e3 = e3[(e3.feasible == True) & np.isclose(e3.alpha, ALPHA)]
        tr, te = make_blocks(D, M, seed); G, Gt = tr["_G"], te["_G"]; I = ~Gt
        rng = np.random.default_rng(2026 + seed)                         # identical to spoof_trait.py
        idx = rng.integers(0, int(Gt.sum()), size=Gt.shape)
        gen_draw = {m: te[m][0][Gt][idx] for m in M}
        sdg = {m: float(tr[m][0][G].std()) for m in M}
        llr = {m: llr_function(tr[m][0][G], tr[m][0][~G]) for m in M}
        mu = {m: (tr[m][0][~G].mean(), tr[m][0][~G].std() + 1e-12) for m in M}
        X_tr = np.stack([tr[m][0].ravel() for m in M], 1); mu_x, sd_x = X_tr.mean(0), X_tr.std(0) + 1e-12
        y = G.ravel(); rg = np.random.default_rng(seed); imp_idx = np.nonzero(~y)[0]
        sub = np.concatenate([np.nonzero(y)[0], rg.choice(imp_idx, min(len(imp_idx), 200000), replace=False)])
        lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit((X_tr[sub] - mu_x) / sd_x, y[sub]); del X_tr
        trait_of = {m: t for t, ms in TR.items() for m in ms}
        # attack models: name -> function(trait) -> score map
        def attacked(trait, model, par):
            out = {}
            for m in M:
                if m not in TR[trait]: out[m] = te[m][0]; continue
                if model == "strength": out[m] = (1.0 - par) * te[m][0] + par * gen_draw[m]
                else: out[m] = gen_draw[m] + par * sdg[m]
            return out
        MODELS = [("strength", l) for l in LAMS] + [("shift", d) for d in SHIFTS]
        for meth in METHODS:
            s = sel[(sel.dataset == ds) & (sel.seed == seed) & np.isclose(sel.alpha, ALPHA) & (sel.method == meth)]
            if len(s) == 0: continue
            orders = s.orders.iloc[0].split("|")
            acc_rates = {(mo, p, t): [] for mo, p in MODELS for t in TR}
            pad_rates = {(a, t): [] for a in APCERS for t in TR}
            frr_pad, acq_g, acq_i, frr0 = [], [], [], []
            for order in orders:
                r = e3[(e3.method == meth) & (e3.order == order)].iloc[0]; thr = json.loads(r.thr)
                par_ = meth.startswith("P")
                chain = list(M) if par_ else order.split(">")
                thr = thr if par_ else [tuple(x) if isinstance(x, list) else x for x in thr]
                def decide(sc):
                    if par_:
                        if meth == "P0-Parallel": f = sum((sc[m] - mu[m][0]) / mu[m][1] for m in M)
                        elif meth == "P1-LLR": f = sum(llr[m](sc[m]) for m in M)
                        else:
                            X = np.stack([sc[m].ravel() for m in M], 1); f = lr.decision_function((X - mu_x) / sd_x).reshape(Gt.shape)
                        return f >= thr[0], np.full(Gt.shape, len(M), np.int8)
                    xs = [sc[m] for m in chain]
                    if meth == "S3-SPRT":
                        c = 0; cum = []
                        for m, x in zip(chain, xs): c = c + llr[m](x); cum.append(c)
                        xs = cum
                    return paths(xs, thr)
                # trait acquisitions as a function of the stages used
                fu = {t: (1 if par_ else first_use(chain, TR[t])) for t in TR}
                def n_acq(used, exclude=None):
                    return sum((used >= fu[t]).astype(np.int16) for t in TR if t != exclude)
                # genuine users and zero-effort impostors (no attack)
                accg, usedg = decide({m: te[m][0] for m in M})
                frr0.append(float((~accg[Gt]).mean()))
                ng = n_acq(usedg); acq_g.append(float(ng[Gt].mean())); acq_i.append(float(ng[I].mean()))
                frr_pad.append(float(1.0 - (accg[Gt] * (1.0 - BPCER) ** ng[Gt]).mean()))
                for t in TR:
                    if not par_ and not any(m in chain for m in TR[t]):
                        for mo, p in MODELS: acc_rates[(mo, p, t)].append(np.nan)
                        for a in APCERS: pad_rates[(a, t)].append(np.nan)
                        continue
                    for mo, p in MODELS:
                        acc, used = decide(attacked(t, mo, p))
                        acc_rates[(mo, p, t)].append(float(acc[I].mean()))
                        if mo == "strength" and p == 1.0:
                            met = used >= fu[t]; nb = n_acq(used, exclude=t)
                            for a in APCERS:
                                pa = acc * np.where(met, a, 1.0) * (1.0 - BPCER) ** nb
                                pad_rates[(a, t)].append(float(pa[I].mean()))
            row = {"dataset": ds, "seed": seed, "alpha": ALPHA, "method": meth, "n_orders": len(orders),
                   "frr_test": float(np.mean(frr0)), "frr_test_pad": float(np.mean(frr_pad)),
                   "acq_gen": float(np.mean(acq_g)), "acq_imp": float(np.mean(acq_i))}
            for mo, p in MODELS:
                per = [np.nanmean(acc_rates[(mo, p, t)]) for t in TR if not np.all(np.isnan(acc_rates[(mo, p, t)]))]
                row[f"{mo}_{p:g}_worst"] = max(per); row[f"{mo}_{p:g}_mean"] = float(np.mean(per))
            for a in APCERS:
                per = [np.nanmean(pad_rates[(a, t)]) for t in TR if not np.all(np.isnan(pad_rates[(a, t)]))]
                row[f"pad_{a:g}_worst"] = max(per); row[f"pad_{a:g}_mean"] = float(np.mean(per))
            rows.append(row)
        print(ds, seed, flush=True)
    del D
    return rows


if __name__ == "__main__":
    dss = sys.argv[1:] or ["fing_x_face", "fing_x_fing", "face_x_face", "lfw_x_fing"]
    for ds in dss:
        pd.DataFrame(run(ds)).to_csv(f"../results/tables/spoof_sens_raw_{ds}.csv", index=False)
    A = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../results/tables/spoof_sens_raw_*.csv"))], ignore_index=True)
    val = [c for c in A.columns if c.endswith("_worst") or c.endswith("_mean") or c in ("frr_test", "frr_test_pad", "acq_gen", "acq_imp")]
    agg = A.groupby(["dataset", "method"])[val].mean().reset_index(); agg["n"] = A.groupby(["dataset", "method"]).size().values
    agg["source"] = "spoof_sens.py: designs of T_rev_selected (E3b) / T_fresh_selected boot (E3fresh, D4) at alpha=1e-3 + test scores"
    agg.to_csv("../results/tables/T_spoof_sens.csv", index=False)
    pd.set_option("display.width", 250)
    print(agg[["dataset", "method"] + [c for c in val if "worst" in c] + ["frr_test", "frr_test_pad", "acq_gen", "acq_imp"]].round(3).to_string())
