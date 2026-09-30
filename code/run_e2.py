"""E2: solver comparison on identical fitting sets (no constraint generation).
Usage: python run_e2.py <dataset> <seed>"""
import sys, os, time, json, numpy as np, pandas as pd
from experiment_core import fitting_data, make_blocks, B_DEFAULT
from envelope_solvers import fit_enum, fit_lr, fit_de, fit_nls_shift, objective_any
from data import MATCHERS, data_file

def main():
    ds, seed = sys.argv[1], int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    D = dict(np.load(data_file(ds))); M = MATCHERS[ds]
    tr, te = make_blocks(D, M, seed); del D; G = tr["_G"]; B = B_DEFAULT
    rows = []; os.makedirs("../results/E2", exist_ok=True)
    for m in M:
        S = tr[m][0]; xf, yf, _, _ = fitting_data(S[G], S[~G])
        for obj in ["P1", "P2"]:
            recs = {}
            t = time.time(); terms, v, info = fit_enum(xf, yf, B, N, obj); recs["ENUM"] = (v, np.nan, time.time() - t, info.get("n_qp"), np.nan)
            for name, mode, ob in [("LR-root (no OBBT)", "root", False), ("LR-root + OBBT", "root", True),
                                   ("LR-BB (no OBBT)", "bb", False), ("LR-BB + OBBT", "bb", True)]:
                t = time.time(); terms, v, info = fit_lr(xf, yf, B, N, obj, mode=mode, obbt=ob, time_limit=1800)
                recs[name] = (v, info["lb"], time.time() - t, info["n_qp"], info["nodes"])
            t = time.time(); terms, v, info = fit_de(xf, yf, N, obj, seed=seed); recs["DE"] = (v, np.nan, time.time() - t, np.nan, np.nan)
            t = time.time(); terms, v = fit_nls_shift(xf, yf, N, obj, seed=seed)
            recs["NLS-shift"] = (v if terms else np.nan, np.nan, time.time() - t, np.nan, np.nan)
            ve = recs["ENUM"][0]
            for k, (v, lb, tt, nqp, nodes) in recs.items():
                rows.append({"dataset": ds, "seed": seed, "matcher": m, "obj": obj, "N": N, "method": k, "value": v, "lb": lb,
                             "rel_to_exact": (v - ve) / ve if ve > 0 else np.nan,
                             "cert_gap": (v - lb) / v if (v > 0 and np.isfinite(lb)) else np.nan,
                             "lb_valid": bool(lb <= ve * (1 + 1e-9) + 1e-12) if np.isfinite(lb) else np.nan,
                             "time": tt, "n_qp": nqp, "nodes": nodes, "n_fit": len(xf)})
            print(f"[{ds} s{seed}] {m} {obj} done", flush=True)
    pd.DataFrame(rows).to_csv(f"../results/E2/e2_{ds}_s{seed}_N{N}.csv", index=False)

if __name__ == "__main__":
    main()
