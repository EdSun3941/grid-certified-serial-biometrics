"""E8 (round-2 revision, reviewer 1): design constraints that only the GP formulation can take.
For the most frequently selected order of each subset (alpha = 1e-3), the proposed GP design is re-solved
with a cap kappa on the expected number of stages of an impostor claim (posynomial 1 + sum prod a_rej <= kappa);
if calibration is infeasible, the GP FAR target is backed off (factors BACKOFF).  Every design receives the same subject-bootstrap calibration as in E3b and is evaluated on the test half: FAR, FRR,
stages per genuine and impostor claim, and the worst-case single-modality spoof acceptance (as in spoof_rev.py).
For comparison, the SPRT on the same order is tuned under the same stage caps (smallest training FRR among its
boundary grid whose training impostor stages do not exceed kappa), and the unconstrained Marcialis rule and parallel
fusion are evaluated once.  Output: results/E8/cost_<dataset>_s<seed>.csv
Usage: python run_cost.py <dataset> <seed>"""
import json, os, sys, numpy as np, pandas as pd
import serial_design as sd
from serial_design import (Envelope, gp_design, calibrated_design, simulate, llr_function, final_threshold_boot,
                           rule_marcialis, _final)
from experiment_core import make_blocks
from data import MATCHERS, data_file, split_subjects

ORDER = {"fing_x_face": ["ri_V", "face_C", "face_G", "li_V"], "fing_x_fing": ["ri_V", "li_V"], "face_x_face": ["face_C", "face_G"]}
ALPHA = 1e-3
KAPPA = {2: [None, 1.8, 1.6, 1.4, 1.2, 1.1], 4: [None, 3.0, 2.5, 2.0, 1.6, 1.3]}
BACKOFF = [1.0, 0.8, 0.64, 0.5, 0.4, 0.3, 0.2]

def spoof_worst(sc_te, Gt, thr, rng, cum_fn=None):
    """Worst-case (over modalities of the chain) acceptance of an impostor presenting a perfect spoof of one modality."""
    I = ~Gt; out = []
    for k in range(len(sc_te)):
        sp = [(rng.choice(sc_te[k][Gt], size=Gt.shape, replace=True) if j == k else sc_te[j]) for j in range(len(sc_te))]
        if cum_fn is not None: sp = cum_fn(sp)
        decided = np.zeros(Gt.shape, bool); acc = np.zeros(Gt.shape, bool)
        for s in range(len(sp)):
            und = ~decided
            if s < len(sp) - 1:
                ta, tr = thr[s]; a = und & (sp[s] >= ta); r = und & (sp[s] < tr); acc |= a; decided |= a | r
            else:
                acc |= und & (sp[s] >= thr[s])
        out.append(float(acc[I].mean()))
    return max(out)

def main():
    ds, seed = sys.argv[1], int(sys.argv[2]); order = ORDER[ds]; S = len(order)
    e1 = pd.read_csv(f"../results/E1/main_{ds}_s{seed}.csv")
    D = dict(np.load(data_file(ds)))
    r_tr, c_tr, _, _ = split_subjects(D["row_subject"], D["col_subject"], seed)
    sd.BOOT.update(row_subj=D["row_subject"][r_tr], col_subj=D["col_subject"][c_tr], B=300, seed=1000 + seed)
    tr, te = make_blocks(D, MATCHERS[ds], seed); del D; G, Gt = tr["_G"], te["_G"]
    envs = []; xd = pd.read_csv(f"../results/E1xd/xd_{ds}_s{seed}.csv")       # exact-dual LR-BB envelopes (Sec. IV)
    for m in order:
        r = e1[(e1.method == "LR-P1-N2") & (e1.matcher == m)].iloc[0]
        t = xd[(xd.method == "LR-P1-N2") & (xd.matcher == m)].iloc[0].terms_new
        envs.append(Envelope(r.kind, [tuple(v) for v in json.loads(t)], r.xmin, r.xmax))
    sc_tr = [tr[m][0] for m in order]; sc_te = [te[m][0] for m in order]
    imps = [np.sort(s[~G]) for s in sc_tr]; gens = [np.sort(s[G]) for s in sc_tr]
    rows = []; rng = np.random.default_rng(100 + seed)
    def record(method, param, thr, pred=np.nan, s_tr=sc_tr, s_te=sc_te, cum_fn=None):
        row = dict(dataset=ds, seed=seed, alpha=ALPHA, order=">".join(order), method=method, param=param, frr_pred_cal=pred)
        if thr is None: row["feasible"] = False; rows.append(row); return
        fa, fr, sg, si = simulate(s_tr, G, thr); row.update(feasible=True, far_train=fa, frr_train=fr, stages_gen_train=sg, stages_imp_train=si)
        fa, fr, sg, si = simulate(s_te, Gt, thr); row.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si)
        row["spoof_worst"] = spoof_worst(sc_te, Gt, thr, rng, cum_fn); row["thr"] = json.dumps([list(t) if isinstance(t, tuple) else t for t in thr])
        rows.append(row); print(method, param, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}, flush=True)
    def gp(**kw):
        """GP design deployed with the E3b calibration; if the early acceptances alone violate the bootstrap
        FAR bound, the GP is re-solved with a smaller FAR target (back-off factor recorded)."""
        for f in BACKOFF:
            d = gp_design(envs, ALPHA * f, **kw)
            if d is None: continue
            c = calibrated_design(d, envs, sc_tr, G, ALPHA, conf=0.95, joint=True, imp_sorted=imps, final="boot")
            if c is not None: return c[0], c[1], f
        return None, np.nan, np.nan
    for kappa in KAPPA[S]:
        thr, pred, f = gp(imp_stage_cap=kappa); record("GP stage cap", kappa if kappa else np.inf, thr, pred); rows[-1]["backoff"] = f
    # SPRT on the same order, tuned under the same impostor-stage caps
    llrs = [llr_function(s[G], s[~G]) for s in sc_tr]
    def cum_fn(sc):
        out, c = [], 0
        for k, s in enumerate(sc): c = c + llrs[k](s); out.append(c)
        return out
    ctr, cte = cum_fn(sc_tr), cum_fn(sc_te)
    lg, li = ctr[0][G], ctr[0][~G]
    A_grid = np.quantile(li, 1 - np.geomspace(max(1.0 / li.size, ALPHA * 1e-2), ALPHA, 12))
    B_grid = np.quantile(lg, np.geomspace(1e-3, 0.3, 12))
    cands = []
    for A in A_grid:
        for Bt in B_grid:
            if Bt >= A: continue
            pre = [(float(A), float(Bt))] * (S - 1); tS = _final(ctr, G, pre, ALPHA, None)
            if tS is None: continue
            fa, fr, sg, si = simulate(ctr, G, pre + [tS]); cands.append((fr, si, pre))
    for kappa in KAPPA[S]:
        ok = sorted([c for c in cands if kappa is None or c[1] <= kappa], key=lambda c: c[0]); thr = None
        for fr, si, pre in ok:                       # best training FRR whose calibration is feasible
            tS = final_threshold_boot(ctr, G, pre, ALPHA, 0.95)
            if tS is not None: thr = pre + [tS]; break
        record("SPRT stage cap", kappa if kappa else np.inf, thr, s_tr=ctr, s_te=cte, cum_fn=cum_fn)
    # unconstrained references
    thr = rule_marcialis(sc_tr, G, ALPHA)
    if thr is not None:
        tS = final_threshold_boot(sc_tr, G, thr[:-1], ALPHA, 0.95); thr = None if tS is None else thr[:-1] + [tS]
    record("Marcialis", np.inf, thr)
    os.makedirs("../results/E8", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"../results/E8/cost_{ds}_s{seed}.csv", index=False)

if __name__ == "__main__":
    main()
