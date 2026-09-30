"""E7 (revision): scalability of the exact-dual LR-BB (P1, P2) and the P2 MILP in N, |D| and |B|.
Usage: python run_e7v2.py <dataset> <matcher> <seed> <N|D|B> [time_limit]"""
import sys, os, time, math, numpy as np, pandas as pd
from experiment_core import fitting_data, make_blocks
from envelope_solvers import fit_exact_dual, fit_milp_p2
from data import data_file

def run(x, y, B, N, tl):
    out = []
    for obj in ["P1", "P2"]:
        t = time.time(); _, v, info = fit_exact_dual(x, y, B, N, obj, time_limit=tl)
        out.append({"obj": obj, "method": "LR-BB exact dual", "value": v, "lb": info["lb"], "time": time.time() - t,
                    "complete": info["complete"], "cert_gap": info["gap"], "root_gap": info["root_gap"], "nodes": info["nodes"]})
    t = time.time(); _, v, info = fit_milp_p2(x, y, B, N, time_limit=tl)
    out.append({"obj": "P2", "method": "MILP (HiGHS)", "value": v, "lb": np.nan, "time": time.time() - t, "complete": info["complete"]})
    return out

def main():
    ds, m, seed, which = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    tl = float(sys.argv[5]) if len(sys.argv) > 5 else 600
    from experiment_core import make_blocks
    D = dict(np.load(data_file(ds))); tr, te = make_blocks(D, [m], seed); del D
    G = tr["_G"]; S = tr[m][0]; rows = []
    Bdef = np.round(np.arange(-3, 1e-9, 0.02), 4)
    if which == "N":
        x, y, _, _ = fitting_data(S[G], S[~G])
        for N in [1, 2, 3, 4, 5]:
            for r in run(x, y, Bdef, N, tl): r.update(N=N, nD=len(x), nB=len(Bdef)); rows.append(r)
            print("N", N, "done", flush=True)
    elif which == "D":
        for nb in [30, 60, 120, 240, 480, 960]:
            x, y, _, _ = fitting_data(S[G], S[~G], nbins=nb)
            for r in run(x, y, Bdef, 2, tl): r.update(N=2, nD=len(x), nB=len(Bdef), nbins=nb); rows.append(r)
            print("nbins", nb, "done", flush=True)
    elif which == "B":
        x, y, _, _ = fitting_data(S[G], S[~G])
        for st in [0.1, 0.05, 0.02, 0.01, 0.005]:
            B = np.round(np.arange(-3, 1e-9, st), 4)
            for r in run(x, y, B, 2, tl): r.update(N=2, nD=len(x), nB=len(B), bstep=st); rows.append(r)
            print("bstep", st, "done", flush=True)
    os.makedirs("../results/E7v2", exist_ok=True)
    df = pd.DataFrame(rows); df.insert(0, "which", which); df.insert(0, "matcher", m); df.insert(0, "dataset", ds)
    df.to_csv(f"../results/E7v2/e7v2_{ds}_{m}_{which}.csv", index=False)

if __name__ == "__main__":
    main()
