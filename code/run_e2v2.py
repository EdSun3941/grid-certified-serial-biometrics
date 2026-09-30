"""E2 (revision): exact-dual LR-BB and the P2 MILP reference on the SAME fitting sets as E2.
Also records the exact root bound with and without the cutoff-based caps, and whether the root closes the gap.
Usage: python run_e2v2.py <dataset> <seed> [N]"""
import sys, os, time, numpy as np, pandas as pd
from experiment_core import fitting_data, make_blocks, B_DEFAULT
from envelope_solvers import fit_exact_dual, fit_milp_p2, exact_dual_node, _design
from data import MATCHERS, data_file

def main():
    ds, seed = sys.argv[1], int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    D = dict(np.load(data_file(ds))); M = MATCHERS[ds]
    tr, te = make_blocks(D, M, seed); del D; G = tr["_G"]; B = B_DEFAULT
    old = pd.read_csv(f"../results/E2/e2_{ds}_s{seed}_N{N}.csv") if N == 2 else None
    ref = pd.read_csv(f"../results/E2ref/ref_{ds}_s{seed}.csv") if N == 2 else None     # independent exact reference
    rows = []; os.makedirs("../results/E2v2", exist_ok=True)
    for m in M:
        S = tr[m][0]; xf, yf, _, _ = fitting_data(S[G], S[~G])
        for obj in ["P1", "P2"]:
            ve = ref[(ref.matcher == m) & (ref.obj == obj)].value.iloc[0] if ref is not None else np.nan
            t = time.time(); terms, v, info = fit_exact_dual(xf, yf, B, N, obj, time_limit=1800); el = time.time() - t
            Phi, yv = _design(xf, yf, B, False)
            zc, _, _, _ = exact_dual_node(Phi, yv, obj, N, [(0, len(B) - 1)] * N, v, caps=False)
            recs = [("LR-BB exact dual", v, info["lb"], el, info["nodes"], info["root_gap"], (v - zc) / v if v > 0 else np.nan, info["complete"])]
            extra = {"root_lb": info["root_lb"], "root_ub": info["root_ub"], "n_open_leaves": info["n_open_leaves"], "ref_value": ve}
            if obj == "P2":
                t = time.time(); terms2, v2, info2 = fit_milp_p2(xf, yf, B, N, time_limit=1800); el2 = time.time() - t
                recs.append(("MILP (HiGHS)", v2, np.nan, el2, np.nan, np.nan, np.nan, info2["complete"]))
            for k, v_, lb, tt, nodes, rg, rg_nocap, comp in recs:
                rows.append({"dataset": ds, "seed": seed, "matcher": m, "obj": obj, "N": N, "method": k, "value": v_, "lb": lb,
                             "rel_to_exact": (v_ - ve) / ve if ve > 0 else np.nan,
                             "cert_gap": (v_ - lb) / v_ if (v_ > 0 and np.isfinite(lb)) else np.nan,
                             "root_gap": rg, "root_gap_nocaps": rg_nocap, "nodes": nodes, "time": tt, "complete": comp, "n_fit": len(xf),
                             **(extra if k == "LR-BB exact dual" else {"ref_value": ve})})
            print(f"[{ds} s{seed}] {m} {obj} done", flush=True)
    pd.DataFrame(rows).to_csv(f"../results/E2v2/e2v2_{ds}_s{seed}_N{N}.csv", index=False)

if __name__ == "__main__":
    main()
