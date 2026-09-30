"""E3-cal: calibrated deployment (CP 95% margins + joint final-stage calibration) for every GP design,
re-using the envelopes fitted in E1 (no refitting), and confidence-calibrated classical rules.
Usage: python run_cal.py <dataset> <seed> [--tag main] [--conf 0.95]"""
import argparse, json, os, time, numpy as np, pandas as pd
from experiment_core import make_blocks, all_orders
from serial_design import (Envelope, gp_design, calibrated_design, simulate, rule_marcialis, rule_symmetric,
                           rule_direct, parallel_sum)
from data import MATCHERS, data_file
ALPHAS = {"fing_x_face": [1e-2, 1e-3], "face_x_face": [1e-2, 1e-3, 1e-4], "fing_x_fing": [1e-2, 1e-3, 1e-4]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed", type=int)
    ap.add_argument("--tag", default="main"); ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--frac", type=float, default=0.5); ap.add_argument("--rules", type=int, default=1)
    a = ap.parse_args()
    e1 = pd.read_csv(f"../results/E1/{a.tag}_{a.dataset}_s{a.seed}.csv")
    D = dict(np.load(data_file(a.dataset))); M = MATCHERS[a.dataset]
    tr, te = make_blocks(D, M, a.seed, a.frac); del D; G, Gt = tr["_G"], te["_G"]
    envs = {}
    for _, r in e1.iterrows():
        envs[(r.method, r.matcher)] = Envelope(r.kind, [tuple(t) for t in json.loads(r.terms)], r.xmin, r.xmax)
    methods = list(e1.method.unique()); rows = []; t0 = time.time()
    IMPS = {m: np.sort(tr[m][0][~G]) for m in M}   # sorted training impostor scores, computed once
    for alpha in ALPHAS[a.dataset]:
        for order in all_orders(M):
            sc_tr = [tr[m][0] for m in order]; sc_te = [te[m][0] for m in order]
            for meth in methods:
                ev = [envs[(meth, m)] for m in order]; t1 = time.time()
                d = gp_design(ev, alpha)
                row = {"dataset": a.dataset, "seed": a.seed, "alpha": alpha, "order": ">".join(order), "n_stages": len(order),
                       "method": meth, "kind": "GP", "tag": a.tag, "conf": a.conf}
                if d is None:
                    row["feasible"] = False; rows.append(row); continue
                c = calibrated_design(d, ev, sc_tr, G, alpha, conf=a.conf, joint=True, imp_sorted=[IMPS[m] for m in order])
                if c is None:
                    row["feasible"] = False; row["gp_feasible"] = True; rows.append(row); continue
                thr, pred = c
                fa, fr, _, _ = simulate(sc_tr, G, thr); row.update(feasible=True, frr_pred_gp=d["frr_pred"], frr_pred_cal=pred,
                                                                  far_train=fa, frr_train=fr)
                fa, fr, sg, si = simulate(sc_te, Gt, thr); row.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si,
                                                                     time=time.time() - t1, thr=json.dumps([list(t) if isinstance(t, tuple) else t for t in thr]))
                rows.append(row)
            if a.rules:
                for rule in ["S1-Marcialis", "S2-Symmetric", "S4-Direct"]:
                    row = {"dataset": a.dataset, "seed": a.seed, "alpha": alpha, "order": ">".join(order), "n_stages": len(order),
                           "method": rule, "kind": "rule", "tag": a.tag, "conf": a.conf}
                    thr = None
                    if rule == "S1-Marcialis" and len(order) >= 2: thr = rule_marcialis(sc_tr, G, alpha, conf=a.conf)
                    elif rule == "S2-Symmetric" and len(order) >= 2: thr = rule_symmetric(sc_tr, G, alpha, conf=a.conf)
                    elif rule == "S4-Direct" and len(order) == 2: thr = rule_direct(sc_tr, G, alpha, 1.0 / (~G).sum(), conf=a.conf)
                    if thr is None: row["feasible"] = False; rows.append(row); continue
                    fa, fr, _, _ = simulate(sc_tr, G, thr); row.update(feasible=True, far_train=fa, frr_train=fr)
                    fa, fr, sg, si = simulate(sc_te, Gt, thr); row.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si)
                    rows.append(row)
        if a.rules:
            t, z_tr, z_te = parallel_sum([tr[m][0] for m in M], G, [te[m][0] for m in M], alpha, conf=a.conf)
            rows.append({"dataset": a.dataset, "seed": a.seed, "alpha": alpha, "order": "parallel", "n_stages": len(M),
                         "method": "P0-Parallel", "kind": "rule", "tag": a.tag, "conf": a.conf, "feasible": True,
                         "far_train": float((z_tr[~G] >= t).mean()), "frr_train": float((z_tr[G] < t).mean()),
                         "far_test": float((z_te[~Gt] >= t).mean()), "frr_test": float((z_te[Gt] < t).mean()),
                         "stages_gen": float(len(M)), "stages_imp": float(len(M))})
        print(f"[cal {a.dataset} s{a.seed}] alpha {alpha} {time.time()-t0:.0f}s", flush=True)
    os.makedirs("../results/E3cal", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"../results/E3cal/{a.tag}_{a.dataset}_s{a.seed}.csv", index=False)

if __name__ == "__main__":
    main()
