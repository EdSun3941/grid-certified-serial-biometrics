"""IJIS v08 (review M4, M5): sensitivity of the final-stage calibration of the selected designs (original splits 0-9 of
D1-D3, results/E3b, selection of results/tables/T_rev_selected.csv including tied orders).

For every selected design the earlier stage thresholds are kept and only the final threshold is recalibrated:
  paper      B = 300 replicates, seed 1000 + split (reproduces the stored design; checked)
  B1000      B = 1000 replicates, seed 1000 + split
  seed7000   B = 300, seed 7000 + split
  seed8000   B = 300, seed 8000 + split
  nofloor    (GP designs only) paper bootstrap without the step-(i) Clopper-Pearson floor, i.e. final threshold = t_boot
             instead of max(t_CP, t_boot)
For GP designs it also records, per stage threshold, the realized training FAR and FRR of the matcher and the envelope
value at that FAR (Lemma 1: FRR_s(t) <= g_s(FAR_s(t)) whenever FAR_s(t) > 0; at FAR 0 the threshold is the most
permissive one and g_s(x_1) applies).
Usage: python calib_sensitivity.py <dataset> <seed>  ->  results/E3sens/sens_<dataset>_s<seed>.csv (+ lemma_*.csv)"""
import json, os, sys, time, numpy as np, pandas as pd
import serial_design as sd
from serial_design import Envelope, gp_design, calibrated_design, simulate, rule_sprt, llr_function, final_threshold_boot
from sklearn.linear_model import LogisticRegression
from experiment_core import make_blocks
from data import MATCHERS, data_file, split_subjects

PRIMARY = ["LR-P1-N2", "HYP", "S1-Marcialis", "S2-Symmetric", "S4-Direct", "S3-SPRT", "P0-Parallel", "P1-LLR", "P2-LogReg"]
VARIANTS = [("paper", 300, 1000), ("B1000", 1000, 1000), ("seed7000", 300, 7000), ("seed8000", 300, 8000)]


def main():
    ds, seed = sys.argv[1], int(sys.argv[2]); t0 = time.time()
    sel = pd.read_csv("../results/tables/T_rev_selected.csv")
    sel = sel[(sel.dataset == ds) & (sel.seed == seed) & sel.method.isin(PRIMARY)]
    e3 = pd.read_csv(f"../results/E3b/main_{ds}_s{seed}.csv"); e1 = pd.read_csv(f"../results/E1/main_{ds}_s{seed}.csv")
    xd = pd.read_csv(f"../results/E1xd/xd_{ds}_s{seed}.csv")
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
    # parallel fusion scores, exactly as in run_cal2.py
    fused = {}
    z_tr = z_te = 0
    for m in M:
        mu, sdv = tr[m][0][~G].mean(), tr[m][0][~G].std() + 1e-12
        z_tr = z_tr + (tr[m][0] - mu) / sdv; z_te = z_te + (te[m][0] - mu) / sdv
    fused["P0-Parallel"] = ([z_tr], [z_te])
    l_tr = l_te = 0
    for m in M:
        f = llr_function(tr[m][0][G], tr[m][0][~G]); l_tr = l_tr + f(tr[m][0]); l_te = l_te + f(te[m][0])
    fused["P1-LLR"] = ([l_tr], [l_te])
    X_tr = np.stack([tr[m][0].ravel() for m in M], 1); X_te = np.stack([te[m][0].ravel() for m in M], 1)
    mu_x, sd_x = X_tr.mean(0), X_tr.std(0) + 1e-12; y = G.ravel()
    rng = np.random.default_rng(seed); imp_idx = np.nonzero(~y)[0]
    sub = np.concatenate([np.nonzero(y)[0], rng.choice(imp_idx, min(len(imp_idx), 200000), replace=False)])
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit((X_tr[sub] - mu_x) / sd_x, y[sub])
    fused["P2-LogReg"] = ([lr.decision_function((X_tr - mu_x) / sd_x).reshape(G.shape)],
                          [lr.decision_function((X_te - mu_x) / sd_x).reshape(Gt.shape)]); del X_tr, X_te
    rows, lem = [], []
    for _, s in sel.iterrows():
        alpha, meth = float(s.alpha), s.method
        for order_s in str(s.orders).split("|"):
            r = e3[(e3.method == meth) & (e3.order == order_s) & np.isclose(e3.alpha, alpha) & (e3.feasible == True)]
            if len(r) != 1: raise RuntimeError(f"{ds} s{seed} {meth} {order_s} {alpha}: {len(r)} rows")
            r = r.iloc[0]; thr = json.loads(r.thr); thr = [tuple(t) if isinstance(t, list) else t for t in thr]
            order = order_s.split(">") if meth not in fused else []
            cp = None
            if meth in fused: sc_tr, sc_te = fused[meth]
            elif meth == "S3-SPRT":
                sc0 = [tr[m][0] for m in order]; o = rule_sprt(sc0, G, alpha); thr_s, llrs = o
                if not all(np.allclose(a_, b_) for a_, b_ in zip(thr_s[:-1], thr[:-1])): raise RuntimeError("SPRT thresholds not reproduced")
                c1 = c2 = 0; sc_tr, sc_te = [], []
                for k, m in enumerate(order):
                    c1 = c1 + llrs[k](tr[m][0]); c2 = c2 + llrs[k](te[m][0]); sc_tr.append(c1); sc_te.append(c2)
            else:
                sc_tr = [tr[m][0] for m in order]; sc_te = [te[m][0] for m in order]
            if r.kind == "GP":
                ev = [envs[(meth, m)] for m in order]; d = gp_design(ev, alpha)
                pre = calibrated_design(d, ev, sc_tr, G, alpha, conf=0.95, joint=False, imp_sorted=[IMPS[m] for m in order])[0]
                if [tuple(x) if isinstance(x, tuple) else x for x in pre[:-1]] != thr[:-1]: raise RuntimeError("GP stage thresholds not reproduced")
                cp = pre[-1]
                # Lemma 1 check at every deployed stage threshold (training data, marginal per matcher)
                for k, m in enumerate(order):
                    ths = [("acc", thr[k][0]), ("rej", thr[k][1])] if k < len(order) - 1 else [("final", thr[k])]
                    for which, t in ths:
                        far = float(1.0 - np.searchsorted(IMPS[m], t, side="left") / len(IMPS[m]))
                        frr = float(np.searchsorted(GENS[m], t, side="left") / len(GENS[m]))
                        g = float(ev[k](max(far, ev[k].xmin)))
                        lem.append(dict(dataset=ds, seed=seed, alpha=alpha, method=meth, order=order_s, stage=k + 1, which=which,
                                        far=far, frr=frr, g=g, holds=frr <= g * (1 + 1e-9) + 1e-12))
            for name, B, sd0 in VARIANTS:
                sd.BOOT.update(row_subj=rs, col_subj=cs, B=B, seed=sd0 + seed)
                tS = final_threshold_boot(sc_tr, G, thr[:-1], alpha, 0.95)
                out = [(name, tS)] + ([("nofloor", tS)] if (r.kind == "GP" and name == "paper") else [])
                for vname, tb in out:
                    row = dict(dataset=ds, seed=seed, alpha=alpha, method=meth, kind=r.kind, order=order_s, n_tied=len(str(s.orders).split("|")),
                               variant=vname, B=B, boot_seed=sd0 + seed, thr_final_stored=thr[-1] if not isinstance(thr[-1], tuple) else np.nan,
                               t_boot=np.nan if tb is None else tb, t_cp=np.nan if cp is None else cp)
                    if tb is None: row["feasible"] = False; rows.append(row); continue
                    tf = max(cp, tb) if (cp is not None and vname != "nofloor") else tb
                    th = list(thr[:-1]) + [tf]
                    fa, fr, sg, si = simulate(sc_tr, G, th); row.update(feasible=True, thr_final=tf, far_train=fa, frr_train=fr,
                                                                       cp_binds=bool(cp is not None and cp > tb))
                    fa, fr, sg, si = simulate(sc_te, Gt, th); row.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si)
                    if vname == "paper":
                        row["reproduced"] = bool(np.isclose(tf, thr[-1], rtol=0, atol=0) and np.isclose(fr, r.frr_test) and np.isclose(fa, r.far_test))
                    rows.append(row)
        print(f"[sens {ds} s{seed}] {meth} {alpha} {time.time() - t0:.0f}s", flush=True)
    os.makedirs("../results/E3sens", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"../results/E3sens/sens_{ds}_s{seed}.csv", index=False)
    pd.DataFrame(lem).to_csv(f"../results/E3sens/lemma_{ds}_s{seed}.csv", index=False)
    print(f"[sens {ds} s{seed}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
