"""IJIS v08 (review M4, M5): Monte Carlo and CP-floor sensitivity of the final-stage calibration with order
re-selection.  For every primary method and EVERY order of the original splits of D1-D3 (as in run_cal2.py, with the
exact-dual LR-BB envelopes of results/E1xd for the proposed design, as in run_cal2_xd.py), the stage thresholds before
the final bootstrap step are computed once and the final threshold is then recalibrated under each variant:
  paper      B = 300 replicates, seed 1000 + split (reproduces results/E3b; checked by analyze_sens.py)
  B1000      B = 1000 replicates, seed 1000 + split
  seed7000   B = 300, seed 7000 + split
  seed8000   B = 300, seed 8000 + split
  nofloor    GP designs only: paper bootstrap without the step-(i) Clopper-Pearson floor (final threshold t_boot
             instead of max(t_CP, t_boot)); all other steps unchanged
analyze_sens.py then selects, per variant, the order with the smallest training FRR (ties averaged) exactly as in the
paper.  For the GP designs it also checks Proposition 1 at every deployed stage threshold of the paper variant
(training data, per matcher): FRR_s(t) <= g_s(FAR_s(t)), with g_s(x_1) when FAR_s(t) = 0.
Usage: python calib_sensitivity_all.py <dataset> <seed>  ->  results/E3sensall/sensall_<dataset>_s<seed>.csv (+ lemma)"""
import json, os, sys, time, numpy as np, pandas as pd
import serial_design as sd
from serial_design import (Envelope, gp_design, calibrated_design, simulate, rule_marcialis, rule_symmetric, rule_direct,
                           rule_direct_cd, rule_sprt, llr_function, final_threshold_boot)
from sklearn.linear_model import LogisticRegression
from experiment_core import make_blocks, all_orders
from data import MATCHERS, data_file, split_subjects
from run_cal2 import ALPHAS, jthr

GP = ["LR-P1-N2", "HYP"]; RULES = ["S1-Marcialis", "S2-Symmetric", "S3-SPRT", "S4-Direct"]
VARIANTS = [("paper", 300, 1000), ("B1000", 1000, 1000), ("seed7000", 300, 7000), ("seed8000", 300, 8000)]


def main():
    ds, seed = sys.argv[1], int(sys.argv[2]); t0 = time.time()
    e1 = pd.read_csv(f"../results/E1/main_{ds}_s{seed}.csv"); xd = pd.read_csv(f"../results/E1xd/xd_{ds}_s{seed}.csv")
    D = dict(np.load(data_file(ds))); M = MATCHERS[ds]
    r_tr, c_tr, _, _ = split_subjects(D["row_subject"], D["col_subject"], seed)
    rs, cs = D["row_subject"][r_tr], D["col_subject"][c_tr]
    tr, te = make_blocks(D, M, seed); del D; G, Gt = tr["_G"], te["_G"]
    envs = {}
    for _, r in e1[e1.method == "HYP"].iterrows():
        envs[("HYP", r.matcher)] = Envelope(r.kind, [tuple(t) for t in json.loads(r.terms)], r.xmin, r.xmax)
    for _, r in xd[xd.method == "LR-P1-N2"].iterrows():
        x = e1[(e1.method == r.method) & (e1.matcher == r.matcher)].iloc[0]
        envs[("LR-P1-N2", r.matcher)] = Envelope("posy", [tuple(t) for t in json.loads(r.terms_new)], x.xmin, x.xmax)
    IMPS = {m: np.sort(tr[m][0][~G]) for m in M}; GENS = {m: np.sort(tr[m][0][G]) for m in M}
    # stage thresholds of the rule designs before the final bootstrap step do not depend on the bootstrap; for designs that
    # are feasible in results/E3b they are read from there (identical to recomputing them, checked on D1 split 0), the
    # others are recomputed with the rule functions
    e3 = pd.read_csv(f"../results/E3b/main_{ds}_s{seed}.csv"); e3 = e3[(e3.kind == "rule") & (e3.feasible == True)]
    E3THR = {(round(r.alpha, 12), r.order, r.method): [tuple(t) if isinstance(t, list) else t for t in json.loads(r.thr)] for _, r in e3.iterrows()}
    LLR1 = {m: llr_function(tr[m][0][G], tr[m][0][~G]) for m in M}          # as inside rule_sprt
    # ---- designs: stage thresholds before the final bootstrap step (independent of the bootstrap)
    des = []
    # parallel fusion scores (independent of alpha), exactly as in run_cal2.py
    z_tr = z_te = 0
    for m in M:
        mu, sdv = tr[m][0][~G].mean(), tr[m][0][~G].std() + 1e-12
        z_tr = z_tr + (tr[m][0] - mu) / sdv; z_te = z_te + (te[m][0] - mu) / sdv
    l_tr = l_te = 0
    for m in M:
        f = llr_function(tr[m][0][G], tr[m][0][~G]); l_tr = l_tr + f(tr[m][0]); l_te = l_te + f(te[m][0])
    X_tr = np.stack([tr[m][0].ravel() for m in M], 1); X_te = np.stack([te[m][0].ravel() for m in M], 1)
    mu_x, sd_x = X_tr.mean(0), X_tr.std(0) + 1e-12; y = G.ravel()
    rng = np.random.default_rng(seed); imp_idx = np.nonzero(~y)[0]
    sub = np.concatenate([np.nonzero(y)[0], rng.choice(imp_idx, min(len(imp_idx), 200000), replace=False)])
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit((X_tr[sub] - mu_x) / sd_x, y[sub])
    lg_tr = lr.decision_function((X_tr - mu_x) / sd_x).reshape(G.shape); lg_te = lr.decision_function((X_te - mu_x) / sd_x).reshape(Gt.shape)
    del X_tr, X_te
    FUSED = [("P0-Parallel", (z_tr, z_te)), ("P1-LLR", (l_tr, l_te)), ("P2-LogReg", (lg_tr, lg_te))]
    SPRT_CUM = {}                                    # cumulative LLR scores per order (the LLRs do not depend on alpha)
    for alpha in ALPHAS[ds]:
        for order in all_orders(M):
            sc_tr = [tr[m][0] for m in order]; sc_te = [te[m][0] for m in order]
            for meth in GP:
                ev = [envs[(meth, m)] for m in order]; d = gp_design(ev, alpha)
                rec = dict(alpha=alpha, order=">".join(order), method=meth, kind="GP", n_stages=len(order))
                if d is None: rec["pre"] = None; des.append(rec); continue
                pre = calibrated_design(d, ev, sc_tr, G, alpha, conf=0.95, joint=False, imp_sorted=[IMPS[m] for m in order])[0]
                rec.update(pre=list(pre[:-1]), t_cp=pre[-1], sc_tr=sc_tr, sc_te=sc_te, ev=ev, mats=list(order)); des.append(rec)
            if len(order) >= 2:
                for rule in RULES:
                    rec = dict(alpha=alpha, order=">".join(order), method=rule, kind="rule", n_stages=len(order)); thr = None
                    stc_tr, stc_te = sc_tr, sc_te
                    key = (round(alpha, 12), ">".join(order), rule)
                    if key in E3THR:
                        thr = E3THR[key]
                        if rule == "S3-SPRT":
                            if order not in SPRT_CUM:
                                c1 = c2 = 0; a_tr, a_te = [], []
                                for m in order:
                                    c1 = c1 + LLR1[m](tr[m][0]); c2 = c2 + LLR1[m](te[m][0]); a_tr.append(c1); a_te.append(c2)
                                SPRT_CUM[order] = (a_tr, a_te)
                            stc_tr, stc_te = SPRT_CUM[order]
                    elif rule == "S1-Marcialis": thr = rule_marcialis(sc_tr, G, alpha)
                    elif rule == "S2-Symmetric": thr = rule_symmetric(sc_tr, G, alpha)
                    elif rule == "S4-Direct":
                        xmin = 1.0 / (~G).sum()
                        thr = rule_direct(sc_tr, G, alpha, xmin) if len(order) == 2 else rule_direct_cd(sc_tr, G, alpha, xmin)
                    else:
                        out = rule_sprt(sc_tr, G, alpha)
                        if out is not None:
                            thr, llrs = out
                            if order not in SPRT_CUM:
                                c1 = c2 = 0; a_tr, a_te = [], []
                                for s_, m in enumerate(order):
                                    c1 = c1 + llrs[s_](tr[m][0]); c2 = c2 + llrs[s_](te[m][0]); a_tr.append(c1); a_te.append(c2)
                                SPRT_CUM[order] = (a_tr, a_te)
                            stc_tr, stc_te = SPRT_CUM[order]
                    rec.update(pre=None if thr is None else list(thr[:-1]), sc_tr=stc_tr, sc_te=stc_te); des.append(rec)
        for name, (f_tr, f_te) in FUSED:
            des.append(dict(alpha=alpha, order="parallel", method=name, kind="parallel", n_stages=len(M), pre=[], sc_tr=[f_tr], sc_te=[f_te]))
    print(f"[sensall {ds} s{seed}] {len(des)} designs {time.time() - t0:.0f}s", flush=True)
    # ---- final threshold under each variant
    rows, lem = [], []
    for name, B, s0 in VARIANTS:
        sd.BOOT.update(row_subj=rs, col_subj=cs, B=B, seed=s0 + seed)
        for rec in des:
            base = dict(dataset=ds, seed=seed, alpha=rec["alpha"], order=rec["order"], method=rec["method"], kind=rec["kind"],
                        n_stages=rec["n_stages"], B=B, boot_seed=s0 + seed)
            if rec["pre"] is None: rows.append(dict(base, variant=name, feasible=False)); continue
            tb = final_threshold_boot(rec["sc_tr"], G, rec["pre"], rec["alpha"], 0.95)
            outs = [(name, tb)] + ([("nofloor", tb)] if (rec["kind"] == "GP" and name == "paper") else [])
            for vname, t in outs:
                row = dict(base, variant=vname, t_boot=np.nan if t is None else t, t_cp=rec.get("t_cp", np.nan))
                if t is None: row["feasible"] = False; rows.append(row); continue
                tf = max(rec["t_cp"], t) if (rec["kind"] == "GP" and vname != "nofloor") else t
                thr = list(rec["pre"]) + [tf]
                fa, fr, sg, si = simulate(rec["sc_tr"], G, thr)
                row.update(feasible=True, thr_final=tf, cp_binds=bool(rec["kind"] == "GP" and rec["t_cp"] > t), far_train=fa, frr_train=fr)
                fa, fr, sg, si = simulate(rec["sc_te"], Gt, thr)
                if rec["kind"] == "parallel": sg = si = float(len(M))
                row.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si, thr=jthr(thr)); rows.append(row)
                if rec["kind"] == "GP" and vname == "paper":          # Proposition 1 at every deployed stage threshold
                    for k, m in enumerate(rec["mats"]):
                        ths = [("acc", thr[k][0]), ("rej", thr[k][1])] if k < len(rec["mats"]) - 1 else [("final", thr[k])]
                        for which, tt in ths:
                            far = float(1.0 - np.searchsorted(IMPS[m], tt, side="left") / len(IMPS[m]))
                            frr = float(np.searchsorted(GENS[m], tt, side="left") / len(GENS[m]))
                            g = float(rec["ev"][k](max(far, rec["ev"][k].xmin)))
                            lem.append(dict(dataset=ds, seed=seed, alpha=rec["alpha"], method=rec["method"], order=rec["order"], stage=k + 1,
                                            which=which, far=far, frr=frr, g=g, holds=frr <= g * (1 + 1e-9) + 1e-12))
        print(f"[sensall {ds} s{seed}] {name} {time.time() - t0:.0f}s", flush=True)
    os.makedirs("../results/E3sensall", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"../results/E3sensall/sensall_{ds}_s{seed}.csv", index=False)
    pd.DataFrame(lem).to_csv(f"../results/E3sensall/lemma_{ds}_s{seed}.csv", index=False)
    print(f"[sensall {ds} s{seed}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
