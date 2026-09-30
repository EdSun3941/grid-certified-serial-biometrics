"""E3b update (round-2 revision, reviewer 2, N2): the GP rows of the LR-BB envelope variants are recomputed with the
envelopes refitted by the exact-dual LR-BB of Sec. IV (results/E1xd), with exactly the same subject-bootstrap
calibration as run_cal2.py; all other rows (other envelope families, serial rules, SPRT, parallel fusion) are kept.
The first-version file is preserved as results/E3b_round1/.  Usage: python run_cal2_xd.py <dataset> <seed>"""
import json, os, shutil, sys, time, numpy as np, pandas as pd
import serial_design as sd
from serial_design import Envelope, gp_design, calibrated_design, simulate
from experiment_core import make_blocks, all_orders
from data import MATCHERS, data_file, split_subjects
from run_cal2 import ALPHAS, jthr

LR = ["LR-P1-N2", "LR-P1-N3", "LR-P2-N2", "LR-P1-N2-rel", "LR-P2-N2-rel"]

def main():
    ds, seed = sys.argv[1], int(sys.argv[2])
    f = f"../results/E3b/main_{ds}_s{seed}.csv"; os.makedirs("../results/E3b_round1", exist_ok=True)
    keep = f"../results/E3b_round1/main_{ds}_s{seed}.csv"
    if not os.path.exists(keep): shutil.copy(f, keep)
    old = pd.read_csv(keep); e1 = pd.read_csv(f"../results/E1/main_{ds}_s{seed}.csv")
    xd = pd.read_csv(f"../results/E1xd/xd_{ds}_s{seed}.csv")
    D = dict(np.load(data_file(ds))); M = MATCHERS[ds]
    r_tr, c_tr, _, _ = split_subjects(D["row_subject"], D["col_subject"], seed)
    sd.BOOT.update(row_subj=D["row_subject"][r_tr], col_subj=D["col_subject"][c_tr], B=300, seed=1000 + seed)
    tr, te = make_blocks(D, M, seed); del D; G, Gt = tr["_G"], te["_G"]
    envs = {}
    for _, r in xd.iterrows():
        x = e1[(e1.method == r.method) & (e1.matcher == r.matcher)].iloc[0]       # same fitted range as E1
        envs[(r.method, r.matcher)] = Envelope("posy", [tuple(t) for t in json.loads(r.terms_new)], x.xmin, x.xmax)
    IMPS = {m: np.sort(tr[m][0][~G]) for m in M}; rows = []
    for alpha in ALPHAS[ds]:
        for order in all_orders(M):
            sc_tr = [tr[m][0] for m in order]; sc_te = [te[m][0] for m in order]
            for meth in LR:
                ev = [envs[(meth, m)] for m in order]; t1 = time.time()
                row = {"dataset": ds, "seed": seed, "alpha": alpha, "order": ">".join(order), "n_stages": len(order), "method": meth,
                       "kind": "GP", "calib": "boot", "B": 300}
                d = gp_design(ev, alpha)
                if d is None: row["feasible"] = False; rows.append(row); continue
                c = calibrated_design(d, ev, sc_tr, G, alpha, conf=0.95, joint=True, imp_sorted=[IMPS[m] for m in order], final="boot")
                if c is None: row.update(feasible=False, gp_feasible=True); rows.append(row); continue
                thr, pred = c; row.update(frr_pred_gp=d["frr_pred"], frr_pred_cal=pred)
                fa, fr, sg, si = simulate(sc_tr, G, thr); row.update(feasible=True, far_train=fa, frr_train=fr)
                fa, fr, sg, si = simulate(sc_te, Gt, thr); row.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si, thr=jthr(thr))
                row["time"] = time.time() - t1; rows.append(row)
    new = pd.concat([old[~old.method.isin(LR)], pd.DataFrame(rows)], ignore_index=True)[old.columns]
    new.to_csv(f, index=False); print(ds, seed, "updated", len(rows), "rows", flush=True)

if __name__ == "__main__":
    main()
