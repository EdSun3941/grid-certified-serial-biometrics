"""IJIS v04 (secondary baseline, reviewer question "why no neural method?"): parallel fusion by a small multilayer
perceptron (MLP).  For each subset and split, an MLP with two hidden layers of 16 ReLU units (Adam, L2 penalty 1e-4,
at most 300 epochs, random_state = split seed) is trained on the standardized match scores of the design data: all
genuine comparisons and up to 200000 impostor comparisons drawn with seed = split seed, class-balanced by sample
weights, exactly as for the logistic-regression fusion.  The pre-sigmoid output (logit) is the fused score, and the
final threshold is set by the same subject bootstrap (B = 300, 95th percentile of the joint training FAR) as for every
other method.  The MLP is an additional, secondary comparison and is not part of the primary Holm family.
Calibrations: 'boot' on the whole training half (bootstrap seed 1000 + split), for all splits; 'xfit' (held-out
calibration; new splits 10-29 and D4 only): MLP and standardization on fold A (fold split seed 5000 + split),
bootstrap threshold on fold B (seed 4000 + split), as in run_fresh.py.
Usage: python run_mlp.py <dataset> <seed>   ->  results/E3mlp/mlp_<dataset>_s<seed>.csv"""
import argparse, json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import serial_design as sd
from serial_design import final_threshold_boot
from data import MATCHERS, data_file, split_subjects

ALPHAS = {"fing_x_face": [1e-2, 1e-3], "face_x_face": [1e-2, 1e-3, 1e-4], "fing_x_fing": [1e-2, 1e-3, 1e-4],
          "lfw_x_fing": [1e-2, 1e-3, 1e-4]}
HIDDEN, MAX_ITER, CHUNK = (16, 16), 300, 1_000_000


def blocks(D, M, rows, cols):
    return {m: D["S_" + m][np.ix_(rows, cols)] for m in M}, D["genuine"][np.ix_(rows, cols)]


def fit_mlp(blk, G, M, seed):
    X = np.stack([blk[m].ravel() for m in M], 1).astype(np.float64); y = G.ravel()
    mu, sdv = X.mean(0), X.std(0) + 1e-12
    rng = np.random.default_rng(seed); imp_idx = np.nonzero(~y)[0]
    sub = np.concatenate([np.nonzero(y)[0], rng.choice(imp_idx, min(len(imp_idx), 200000), replace=False)])
    ys = y[sub]; w = np.where(ys, len(ys) / (2.0 * ys.sum()), len(ys) / (2.0 * (~ys).sum()))    # 'balanced' weights
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        net = MLPClassifier(hidden_layer_sizes=HIDDEN, activation="relu", solver="adam", alpha=1e-4,
                            max_iter=MAX_ITER, random_state=seed).fit((X[sub] - mu) / sdv, ys, sample_weight=w)
    return net, mu, sdv, time.time() - t0


def logit(net, mu, sdv, blk, M):
    """Pre-sigmoid output of the network, evaluated in chunks (the design halves hold up to 9e6 comparisons)."""
    shape = blk[M[0]].shape; X = np.stack([blk[m].ravel() for m in M], 1).astype(np.float64); out = np.empty(len(X))
    for i in range(0, len(X), CHUNK):
        h = (X[i:i + CHUNK] - mu) / sdv
        for k, (W, b) in enumerate(zip(net.coefs_, net.intercepts_)):
            h = h @ W + b
            if k < len(net.coefs_) - 1: h = np.maximum(h, 0.0)
        out[i:i + CHUNK] = h.ravel()
    return out.reshape(shape)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed", type=int)
    ap.add_argument("--B", type=int, default=300); ap.add_argument("--conf", type=float, default=0.95)
    a = ap.parse_args(); ds, seed = a.dataset, a.seed; M = MATCHERS[ds]
    D = dict(np.load(data_file(ds)))
    r_tr, c_tr, r_te, c_te = split_subjects(D["row_subject"], D["col_subject"], seed)
    Btr, Gtr = blocks(D, M, r_tr, c_tr); Bte, Gte = blocks(D, M, r_te, c_te)
    subj_r, subj_c = D["row_subject"], D["col_subject"]; rows = []
    base = dict(dataset=ds, seed=seed, method="P3-MLP", kind="parallel", n_stages=len(M), B=a.B)

    def evaluate(calib, f_des, G_des, f_te, extra):
        for alpha in ALPHAS[ds]:
            row = dict(base, alpha=alpha, calib=calib, **extra)
            t = final_threshold_boot([f_des], G_des, [], alpha, a.conf)
            if t is None: row["feasible"] = False; rows.append(row); continue
            row.update(feasible=True, far_train=float((f_des[~G_des] >= t).mean()), frr_train=float((f_des[G_des] < t).mean()),
                       far_test=float((f_te[~Gte] >= t).mean()), frr_test=float((f_te[Gte] < t).mean()),
                       stages_gen=float(len(M)), stages_imp=float(len(M)), thr=json.dumps([t]))
            rows.append(row)

    # (a) bootstrap calibration on the whole training half
    net, mu, sdv, ft = fit_mlp(Btr, Gtr, M, seed)
    sd.BOOT.update(row_subj=subj_r[r_tr], col_subj=subj_c[c_tr], B=a.B, seed=1000 + seed)
    evaluate("boot", logit(net, mu, sdv, Btr, M), Gtr, logit(net, mu, sdv, Bte, M), dict(n_iter=net.n_iter_, fit_time=ft))
    # (b) held-out calibration (new splits and D4): model on fold A, final threshold on fold B
    if seed >= 10 or ds == "lfw_x_fing":
        subj = np.unique(subj_c[c_tr]); perm = np.random.default_rng(5000 + seed).permutation(len(subj))
        A = set(subj[perm[: len(subj) // 2]].tolist())
        inA_r = np.array([s in A for s in subj_r[r_tr]]); inA_c = np.array([s in A for s in subj_c[c_tr]])
        rA, cA, rB, cB = r_tr[inA_r], c_tr[inA_c], r_tr[~inA_r], c_tr[~inA_c]
        BA, GA = blocks(D, M, rA, cA); BB, GB = blocks(D, M, rB, cB)
        netA, muA, sdA, ftA = fit_mlp(BA, GA, M, seed)
        sd.BOOT.update(row_subj=subj_r[rB], col_subj=subj_c[cB], B=a.B, seed=4000 + seed)
        evaluate("xfit", logit(netA, muA, sdA, BB, M), GB, logit(netA, muA, sdA, Bte, M), dict(n_iter=netA.n_iter_, fit_time=ftA))
    os.makedirs("../results/E3mlp", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"../results/E3mlp/mlp_{ds}_s{seed}.csv", index=False)
    print(f"[mlp {ds} s{seed}] done", flush=True)


if __name__ == "__main__":
    main()
