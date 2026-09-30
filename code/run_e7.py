"""E7: scalability (time vs N, |D|, |B|) for LR-BB (+OBBT), exact enumeration and DE.
Usage: python run_e7.py <dataset> <matcher> <seed> <which: N|D|B> [time_limit]"""
import sys, os, time, math, numpy as np, pandas as pd
from experiment_core import fitting_data, make_blocks
from envelope_solvers import fit_enum, fit_lr, fit_de
from data import data_file

def run(x, y, B, N, obj, tl, seed):
    out = []
    t = time.time(); _, v, info = fit_lr(x, y, B, N, obj, mode="bb", obbt=True, time_limit=tl)
    out.append({"method": "LR-BB + OBBT", "value": v, "lb": info["lb"], "time": time.time() - t,
                "complete": info.get("complete", True), "cert_gap": info["gap"]})
    total = math.comb(len(B), N)
    t = time.time(); _, v, info = fit_enum(x, y, B, N, obj, time_limit=tl); el = time.time() - t
    frac = info["n_qp"] / total
    out.append({"method": "ENUM", "value": v, "lb": np.nan, "time": el, "complete": info["complete"],
                "enum_fraction": frac, "enum_time_extrapolated": el / max(frac, 1e-12)})
    t = time.time(); _, v, info = fit_de(x, y, N, obj, seed=seed)
    out.append({"method": "DE", "value": v, "lb": np.nan, "time": time.time() - t, "complete": True})
    return out

def main():
    ds, m, seed, which = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    tl = float(sys.argv[5]) if len(sys.argv) > 5 else 900
    D = dict(np.load(data_file(ds))); tr, te = make_blocks(D, [m], seed); del D
    G = tr["_G"]; S = tr[m][0]; rows = []; obj = "P1"
    Bdef = np.round(np.arange(-3, 1e-9, 0.02), 4)
    if which == "N":
        x, y, _, _ = fitting_data(S[G], S[~G])
        for N in [1, 2, 3, 4, 5]:
            for r in run(x, y, Bdef, N, obj, tl, seed): r.update(N=N, nD=len(x), nB=len(Bdef)); rows.append(r)
            print("N", N, "done", flush=True)
    elif which == "D":
        for nb in [30, 60, 120, 240, 480, 960]:
            x, y, _, _ = fitting_data(S[G], S[~G], nbins=nb)
            for r in run(x, y, Bdef, 2, obj, tl, seed): r.update(N=2, nD=len(x), nB=len(Bdef), nbins=nb); rows.append(r)
            print("nbins", nb, "|D|", len(x), "done", flush=True)
    elif which == "B":
        x, y, _, _ = fitting_data(S[G], S[~G])
        for st in [0.1, 0.05, 0.02, 0.01, 0.005]:
            B = np.round(np.arange(-3, 1e-9, st), 4)
            for r in run(x, y, B, 2, obj, tl, seed): r.update(N=2, nD=len(x), nB=len(B), bstep=st); rows.append(r)
            print("bstep", st, "done", flush=True)
    os.makedirs("../results/E7", exist_ok=True)
    df = pd.DataFrame(rows); df.insert(0, "which", which); df.insert(0, "matcher", m); df.insert(0, "dataset", ds)
    df.to_csv(f"../results/E7/e7_{ds}_{m}_{which}.csv", index=False)

if __name__ == "__main__":
    main()
