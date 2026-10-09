"""IJIS v09 (review M1, control arm): fold-A-only calibration on the held-out splits of run_fresh.py.
The training subjects of each split are divided into fold A and fold B exactly as in run_fresh.py (by subject, seed
5000 + split), and every design of the primary family is built on fold A with the same seeds (design_all of
run_fresh.py, bootstrap seed 3000 + split).  Unlike the held-out (xfit) arm, the final threshold is NOT recalibrated
on fold B: the fold-A calibration is deployed as it is.  Fold B is not used at all.  The arm therefore has the same
design data, the same order selection and the same calibration rule as the held-out arm, and differs from it only
in calibrating on the design data; against the paper's protocol (boot) it differs only in using half of the training
subjects.
Reproduction: far_foldA / frr_foldA must equal those of the xfit rows of results/E3fresh (checked by analyze_foldA.py).
Usage: python run_foldA.py <dataset> <seed> [--B 300] [--conf 0.95]
Output: results/E3foldA/foldA_<dataset>_s<seed>.csv (one row per alpha, order and method)"""
import argparse, os, time, numpy as np, pandas as pd
from run_fresh import design_all, blocks, ALPHAS, jthr
from serial_design import simulate
from data import MATCHERS, data_file, split_subjects


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed", type=int)
    ap.add_argument("--B", type=int, default=300); ap.add_argument("--conf", type=float, default=0.95)
    a = ap.parse_args(); ds, seed = a.dataset, a.seed; M = MATCHERS[ds]; alphas = ALPHAS[ds]
    t0 = time.time()
    D = dict(np.load(data_file(ds)))
    r_tr, c_tr, r_te, c_te = split_subjects(D["row_subject"], D["col_subject"], seed)
    Bte, Gte = blocks(D, M, r_te, c_te)
    subj = np.unique(D["col_subject"][c_tr]); perm = np.random.default_rng(5000 + seed).permutation(len(subj))
    A = set(subj[perm[: len(subj) // 2]].tolist())
    inA_r = np.array([s in A for s in D["row_subject"][r_tr]]); inA_c = np.array([s in A for s in D["col_subject"][c_tr]])
    rA, cA = r_tr[inA_r], c_tr[inA_c]
    BA, GA = blocks(D, M, rA, cA)
    subj_r, subj_c = D["row_subject"], D["col_subject"]; del D
    desA, _ = design_all(BA, GA, subj_r[rA], subj_c[cA], M, alphas, seed, a.B, a.conf, 3000 + seed)
    rows = []
    for rec in desA:
        row = {"dataset": ds, "seed": seed, "alpha": rec["alpha"], "order": ">".join(rec["order"]),
               "n_stages": len(M) if rec["kind"] == "parallel" else len(rec["order"]), "method": rec["method"],
               "kind": rec["kind"], "calib": "foldA", "B": a.B}
        if not rec.get("feasible"): row["feasible"] = False; rows.append(row); continue
        thr = rec["thr"]
        fa, fr, _, _ = simulate(rec["smap"](BA), GA, thr); row.update(feasible=True, far_foldA=fa, frr_foldA=fr)
        fa, fr, sg, si = simulate(rec["smap"](Bte), Gte, thr)
        if rec["kind"] == "parallel": sg = si = float(len(M))
        row.update(far_test=fa, frr_test=fr, stages_gen=sg, stages_imp=si, thr=jthr(thr)); rows.append(row)
    os.makedirs("../results/E3foldA", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"../results/E3foldA/foldA_{ds}_s{seed}.csv", index=False)
    print(f"[foldA {ds} s{seed}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
