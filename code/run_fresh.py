"""Fresh-seed replication (IJIS revision, reviewer 3): the primary comparison on NEW subject splits (seeds 10-29,
never used before), with two calibrations fixed before these runs were started:
  boot : the calibration of the paper -- design, stage thresholds and order selection on the whole training half,
         final threshold by the subject bootstrap of the joint training FAR on the whole training half;
  xfit : cross-fitted calibration -- the training subjects are split 50/50 (by subject, random seed 5000 + split)
         into a design fold A and a calibration fold B.  Envelopes, GP designs, rule tuning, LLR/fusion models,
         stage thresholds, and the fold-A bootstrap calibration (used only to select the order) use fold A; the final
         threshold of every design is then re-calibrated by the same subject bootstrap on fold B alone (earlier stage
         thresholds fixed), so that the FAR requirement is checked on subjects not used for any design choice.
Methods: the primary family of the paper -- proposed exact-dual LR-BB envelope (P1, N = 2), HYP, Marcialis rule,
symmetric rejection, direct search, SPRT, and sum / LLR / logistic-regression parallel fusion.
Usage: python run_fresh.py <dataset> <seed> [--B 300] [--conf 0.95]
Output: results/E3fresh/fresh_<dataset>_s<seed>.csv (one row per alpha, order, method and calibration)"""
import argparse, json, os, time, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from experiment_core import make_blocks, all_orders, fit_envelope
import serial_design as sd
from serial_design import (gp_design, calibrated_design, simulate, rule_marcialis, rule_symmetric, rule_direct,
                           rule_direct_cd, rule_sprt, llr_function, final_threshold_boot)
from data import MATCHERS, data_file, split_subjects
ALPHAS = {"fing_x_face": [1e-2, 1e-3], "face_x_face": [1e-2, 1e-3, 1e-4], "fing_x_fing": [1e-2, 1e-3, 1e-4],
          "lfw_x_fing": [1e-2, 1e-3, 1e-4]}
GP_METHODS = {"LR-P1-N2": "XD-P1-N2", "HYP": "HYP"}      # proposed envelope = exact-dual LR-BB of Sect. 4


def jthr(thr):
    return json.dumps([list(t) if isinstance(t, tuple) else t for t in thr])


def design_all(blk, G, rsub, csub, M, alphas, seed, B, conf, boot_seed):
    """All designs of the primary family on the design data `blk` (matcher -> score matrix, genuine mask G).
    Returns a list of design dicts with the calibrated thresholds, the design-data FRR/FAR, the thresholds before the
    final bootstrap step (pre), and a score map that turns any block of the same matchers into stage scores."""
    sd.BOOT.update(row_subj=rsub, col_subj=csub, B=B, seed=boot_seed)
    envs, fit_info = {}, {}
    for meth, fm in GP_METHODS.items():
        for m in M:
            S = blk[m]; env, info = fit_envelope(fm, S[G], S[~G], seed=seed, time_limit=300)
            envs[(meth, m)] = env; fit_info[(meth, m)] = info.get("total_time", np.nan)
    IMPS = {m: np.sort(blk[m][~G]) for m in M}
    # parallel fusion models (fitted once on the design data)
    mus = {m: (blk[m][~G].mean(), blk[m][~G].std() + 1e-12) for m in M}
    llr = {m: llr_function(blk[m][G], blk[m][~G]) for m in M}
    X = np.stack([blk[m].ravel() for m in M], 1); mu_x, sd_x = X.mean(0), X.std(0) + 1e-12; y = G.ravel()
    rng = np.random.default_rng(seed); imp_idx = np.nonzero(~y)[0]
    sub = np.concatenate([np.nonzero(y)[0], rng.choice(imp_idx, min(len(imp_idx), 200000), replace=False)])
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit((X[sub] - mu_x) / sd_x, y[sub]); del X
    fus = {"P0-Parallel": lambda b: [sum((b[m] - mus[m][0]) / mus[m][1] for m in M)],
           "P1-LLR": lambda b: [sum(llr[m](b[m]) for m in M)],
           "P2-LogReg": lambda b: [lr.decision_function((np.stack([b[m].ravel() for m in M], 1) - mu_x) / sd_x)
                                   .reshape(b[M[0]].shape)]}
    out = []
    raw = lambda order: (lambda b: [b[m] for m in order])
    for alpha in alphas:
        for order in all_orders(M):
            sc = [blk[m] for m in order]
            for meth in GP_METHODS:
                ev = [envs[(meth, m)] for m in order]; d = gp_design(ev, alpha)
                rec = dict(alpha=alpha, order=order, method=meth, kind="GP", smap=raw(order))
                if d is None: rec["feasible"] = False; out.append(rec); continue
                imps = [IMPS[m] for m in order]
                c = calibrated_design(d, ev, sc, G, alpha, conf=conf, joint=True, imp_sorted=imps, final="boot")
                pre = calibrated_design(d, ev, sc, G, alpha, conf=conf, joint=False, imp_sorted=imps)[0]  # step (i) only
                if c is None: rec.update(feasible=False, pre=pre); out.append(rec); continue
                rec.update(feasible=True, thr=c[0], pre=pre, frr_pred_cal=c[1]); out.append(rec)
            if len(order) >= 2:
                for rule in ["S1-Marcialis", "S2-Symmetric", "S3-SPRT", "S4-Direct"]:
                    rec = dict(alpha=alpha, order=order, method=rule, kind="rule", smap=raw(order)); thr = None; ssc = sc
                    if rule == "S1-Marcialis": thr = rule_marcialis(sc, G, alpha)
                    elif rule == "S2-Symmetric": thr = rule_symmetric(sc, G, alpha)
                    elif rule == "S4-Direct":
                        xmin = 1.0 / (~G).sum()
                        thr = rule_direct(sc, G, alpha, xmin) if len(order) == 2 else rule_direct_cd(sc, G, alpha, xmin)
                    else:
                        o = rule_sprt(sc, G, alpha)
                        if o is not None:
                            thr, llrs = o
                            def smap(b, order=order, llrs=llrs):
                                c, cum = 0, []
                                for s_, m in enumerate(order): c = c + llrs[s_](b[m]); cum.append(c)
                                return cum
                            rec["smap"] = smap; ssc = smap(blk)
                    if thr is None: rec["feasible"] = False; out.append(rec); continue
                    rec["pre"] = list(thr)
                    tS = final_threshold_boot(ssc, G, thr[:-1], alpha, conf)
                    if tS is None: rec["feasible"] = False; out.append(rec); continue
                    rec.update(feasible=True, thr=list(thr[:-1]) + [tS]); out.append(rec)
        for name, f in fus.items():
            rec = dict(alpha=alpha, order=("parallel",), method=name, kind="parallel", smap=f, pre=[None])
            t = final_threshold_boot(f(blk), G, [], alpha, conf)
            if t is None: rec["feasible"] = False; out.append(rec); continue
            rec.update(feasible=True, thr=[t]); out.append(rec)
    return out, fit_info


def blocks(D, M, rows, cols):
    b = {m: D["S_" + m][np.ix_(rows, cols)] for m in M}
    return b, D["genuine"][np.ix_(rows, cols)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed", type=int)
    ap.add_argument("--B", type=int, default=300); ap.add_argument("--conf", type=float, default=0.95)
    a = ap.parse_args(); ds, seed = a.dataset, a.seed; M = MATCHERS[ds]; alphas = ALPHAS[ds]
    t0 = time.time()
    D = dict(np.load(data_file(ds)))
    r_tr, c_tr, r_te, c_te = split_subjects(D["row_subject"], D["col_subject"], seed)
    Btr, Gtr = blocks(D, M, r_tr, c_tr); Bte, Gte = blocks(D, M, r_te, c_te)
    # fold split of the training subjects (by subject)
    subj = np.unique(D["col_subject"][c_tr]); perm = np.random.default_rng(5000 + seed).permutation(len(subj))
    A = set(subj[perm[: len(subj) // 2]].tolist())
    inA_r = np.array([s in A for s in D["row_subject"][r_tr]]); inA_c = np.array([s in A for s in D["col_subject"][c_tr]])
    rA, cA, rB, cB = r_tr[inA_r], c_tr[inA_c], r_tr[~inA_r], c_tr[~inA_c]
    BA, GA = blocks(D, M, rA, cA); BB, GB = blocks(D, M, rB, cB)
    subj_r, subj_c = D["row_subject"], D["col_subject"]
    rows = []
    base = lambda rec, calib: {"dataset": ds, "seed": seed, "alpha": rec["alpha"], "order": ">".join(rec["order"]),
                               "n_stages": len(M) if rec["kind"] == "parallel" else len(rec["order"]),
                               "method": rec["method"], "kind": rec["kind"], "calib": calib, "B": a.B}
    def evaluate(row, rec, thr, Btrain, Gtrain):
        s_tr = rec["smap"](Btrain); s_te = rec["smap"](Bte)
        fa, fr, sg, si = simulate(s_tr, Gtrain, thr); row.update(feasible=True, far_train=fa, frr_train=fr)
        fa, fr, sg, si = simulate(s_te, Gte, thr); row.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si, thr=jthr(thr))
    # (a) paper calibration on the whole training half
    des, fi = design_all(Btr, Gtr, subj_r[r_tr], subj_c[c_tr], M, alphas, seed, a.B, a.conf, 1000 + seed)
    for rec in des:
        row = base(rec, "boot")
        if not rec.get("feasible"): row["feasible"] = False; rows.append(row); continue
        evaluate(row, rec, rec["thr"], Btr, Gtr); rows.append(row)
    print(f"[fresh {ds} s{seed}] boot done {time.time()-t0:.0f}s", flush=True)
    # (b) cross-fitted: design on fold A, final threshold re-calibrated on fold B
    desA, fiA = design_all(BA, GA, subj_r[rA], subj_c[cA], M, alphas, seed, a.B, a.conf, 3000 + seed)
    sd.BOOT.update(row_subj=subj_r[rB], col_subj=subj_c[cB], B=a.B, seed=4000 + seed)
    for rec in desA:
        row = base(rec, "xfit")
        if not rec.get("feasible"): row["feasible"] = False; rows.append(row); continue
        sA = rec["smap"](BA); fa, fr, _, _ = simulate(sA, GA, rec["thr"]); row.update(far_foldA=fa, frr_foldA=fr)
        sB = rec["smap"](BB); pre = rec["pre"]
        if rec["kind"] == "parallel":
            t = final_threshold_boot(sB, GB, [], rec["alpha"], a.conf); thr = None if t is None else [t]
        else:
            t = final_threshold_boot(sB, GB, rec["thr"][:-1], rec["alpha"], a.conf)
            if t is None: thr = None
            elif rec["kind"] == "GP": thr = list(rec["thr"][:-1]) + [max(pre[-1], t)]     # same form as step (ii)
            else: thr = list(rec["thr"][:-1]) + [t]
        if thr is None: row.update(feasible=False, feasible_foldA=True); rows.append(row); continue
        fa, fr, _, _ = simulate(sB, GB, thr); row.update(far_foldB=fa, frr_foldB=fr)
        evaluate(row, rec, thr, Btr, Gtr); rows.append(row)
    print(f"[fresh {ds} s{seed}] xfit done {time.time()-t0:.0f}s", flush=True)
    os.makedirs("../results/E3fresh", exist_ok=True)
    df = pd.DataFrame(rows); df["fit_time_full"] = np.nansum(list(fi.values())); df["fit_time_foldA"] = np.nansum(list(fiA.values()))
    df.to_csv(f"../results/E3fresh/fresh_{ds}_s{seed}.csv", index=False)


if __name__ == "__main__":
    main()
