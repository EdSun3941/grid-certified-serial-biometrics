"""E3b (revision): calibrated deployment with a SUBJECT-LEVEL bootstrap of the joint training FAR for every
method, plus the additional baselines requested in review (SPRT on all subsets, direct search for chains of
any length, likelihood-ratio and logistic-regression parallel fusion, continuous-exponent monomial).
GP envelopes are re-used from E1 (no refitting), except MONO-cont, which is fitted here.
Usage: python run_cal2.py <dataset> <seed> [--B 300] [--conf 0.95]"""
import argparse, json, os, time, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from experiment_core import make_blocks, all_orders, fit_envelope
import serial_design as sd
from serial_design import (Envelope, gp_design, calibrated_design, simulate, rule_marcialis, rule_symmetric,
                           rule_direct, rule_direct_cd, rule_sprt, llr_function, final_threshold_boot)
from data import MATCHERS, data_file, split_subjects
ALPHAS = {"fing_x_face": [1e-2, 1e-3], "face_x_face": [1e-2, 1e-3, 1e-4], "fing_x_fing": [1e-2, 1e-3, 1e-4]}

def jthr(thr):
    return json.dumps([list(t) if isinstance(t, tuple) else t for t in thr])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed", type=int)
    ap.add_argument("--tag", default="main"); ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--B", type=int, default=300)
    a = ap.parse_args()
    e1 = pd.read_csv(f"../results/E1/{a.tag}_{a.dataset}_s{a.seed}.csv")
    D = dict(np.load(data_file(a.dataset))); M = MATCHERS[a.dataset]
    r_tr, c_tr, _, _ = split_subjects(D["row_subject"], D["col_subject"], a.seed)
    sd.BOOT.update(row_subj=D["row_subject"][r_tr], col_subj=D["col_subject"][c_tr], B=a.B, seed=1000 + a.seed)
    tr, te = make_blocks(D, M, a.seed); del D; G, Gt = tr["_G"], te["_G"]
    envs = {}
    for _, r in e1.iterrows():
        envs[(r.method, r.matcher)] = Envelope(r.kind, [tuple(t) for t in json.loads(r.terms)], r.xmin, r.xmax)
    for m in M:                                             # continuous-exponent monomial (review request)
        S = tr[m][0]; env, _ = fit_envelope("MONO-cont", S[G], S[~G], seed=a.seed); envs[("MONO-cont", m)] = env
    methods = list(e1.method.unique()) + ["MONO-cont"]; rows = []; t0 = time.time()
    IMPS = {m: np.sort(tr[m][0][~G]) for m in M}
    base = lambda alpha, order, meth, kind: {"dataset": a.dataset, "seed": a.seed, "alpha": alpha, "order": ">".join(order),
                                             "n_stages": len(order), "method": meth, "kind": kind, "calib": "boot", "B": a.B}
    def finish(row, thr, sc_tr, sc_te):
        fa, fr, sg, si = simulate(sc_tr, G, thr); row.update(feasible=True, far_train=fa, frr_train=fr)
        fa, fr, sg, si = simulate(sc_te, Gt, thr); row.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si, thr=jthr(thr))
    for alpha in ALPHAS[a.dataset]:
        for order in all_orders(M):
            sc_tr = [tr[m][0] for m in order]; sc_te = [te[m][0] for m in order]
            for meth in methods:
                ev = [envs[(meth, m)] for m in order]; t1 = time.time(); row = base(alpha, order, meth, "GP")
                d = gp_design(ev, alpha)
                if d is None: row["feasible"] = False; rows.append(row); continue
                c = calibrated_design(d, ev, sc_tr, G, alpha, conf=a.conf, joint=True, imp_sorted=[IMPS[m] for m in order], final="boot")
                if c is None: row.update(feasible=False, gp_feasible=True); rows.append(row); continue
                thr, pred = c; row.update(frr_pred_gp=d["frr_pred"], frr_pred_cal=pred)
                finish(row, thr, sc_tr, sc_te); row["time"] = time.time() - t1; rows.append(row)
            if len(order) >= 2:                                   # serial rules: search on point estimates, then calibrate
                for rule in ["S1-Marcialis", "S2-Symmetric", "S3-SPRT", "S4-Direct"]:
                    row = base(alpha, order, rule, "rule"); t1 = time.time(); thr = None; stc_tr, stc_te = sc_tr, sc_te
                    if rule == "S1-Marcialis": thr = rule_marcialis(sc_tr, G, alpha)
                    elif rule == "S2-Symmetric": thr = rule_symmetric(sc_tr, G, alpha)
                    elif rule == "S4-Direct":
                        xmin = 1.0 / (~G).sum()
                        thr = rule_direct(sc_tr, G, alpha, xmin) if len(order) == 2 else rule_direct_cd(sc_tr, G, alpha, xmin)
                    else:
                        out = rule_sprt(sc_tr, G, alpha)
                        if out is not None:
                            thr, llrs = out; c1 = c2 = 0; stc_tr, stc_te = [], []
                            for s_, m in enumerate(order):
                                c1 = c1 + llrs[s_](tr[m][0]); c2 = c2 + llrs[s_](te[m][0]); stc_tr.append(c1); stc_te.append(c2)
                    if thr is None: row["feasible"] = False; rows.append(row); continue
                    tS = final_threshold_boot(stc_tr, G, thr[:-1], alpha, a.conf)
                    if tS is None: row["feasible"] = False; rows.append(row); continue
                    thr = list(thr[:-1]) + [tS]; finish(row, thr, stc_tr, stc_te); row["time"] = time.time() - t1; rows.append(row)
        # parallel fusion of all matchers (single threshold, calibrated in the same way)
        fused = {}
        z_tr = 0; z_te = 0
        for m in M:
            mu, sdv = tr[m][0][~G].mean(), tr[m][0][~G].std() + 1e-12
            z_tr = z_tr + (tr[m][0] - mu) / sdv; z_te = z_te + (te[m][0] - mu) / sdv
        fused["P0-Parallel"] = (z_tr, z_te)
        l_tr = 0; l_te = 0
        for m in M:
            f = llr_function(tr[m][0][G], tr[m][0][~G]); l_tr = l_tr + f(tr[m][0]); l_te = l_te + f(te[m][0])
        fused["P1-LLR"] = (l_tr, l_te)
        X_tr = np.stack([tr[m][0].ravel() for m in M], 1); X_te = np.stack([te[m][0].ravel() for m in M], 1)
        mu_x, sd_x = X_tr.mean(0), X_tr.std(0) + 1e-12; y = G.ravel()
        rng = np.random.default_rng(a.seed); imp_idx = np.nonzero(~y)[0]
        sub = np.concatenate([np.nonzero(y)[0], rng.choice(imp_idx, min(len(imp_idx), 200000), replace=False)])
        lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit((X_tr[sub] - mu_x) / sd_x, y[sub])
        fused["P2-LogReg"] = (lr.decision_function((X_tr - mu_x) / sd_x).reshape(G.shape),
                              lr.decision_function((X_te - mu_x) / sd_x).reshape(Gt.shape))
        for name, (f_tr, f_te) in fused.items():
            row = base(alpha, ["parallel"], name, "parallel"); row["n_stages"] = len(M)
            t = final_threshold_boot([f_tr], G, [], alpha, a.conf)
            if t is None: row["feasible"] = False; rows.append(row); continue
            row.update(feasible=True, far_train=float((f_tr[~G] >= t).mean()), frr_train=float((f_tr[G] < t).mean()),
                       far_test=float((f_te[~Gt] >= t).mean()), frr_test=float((f_te[Gt] < t).mean()),
                       stages_gen=float(len(M)), stages_imp=float(len(M)), thr=json.dumps([t]))
            rows.append(row)
        print(f"[cal2 {a.dataset} s{a.seed}] alpha {alpha} {time.time()-t0:.0f}s", flush=True)
    os.makedirs("../results/E3b", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"../results/E3b/{a.tag}_{a.dataset}_s{a.seed}.csv", index=False)

if __name__ == "__main__":
    main()
