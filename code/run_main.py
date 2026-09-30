"""E1 (envelope fitting) + E3 (system design) + E4-noise, per dataset and seed.
Usage: python run_main.py <dataset> <seed> [--frac 0.5] [--methods ...] [--tag main]
Writes results/E1/<tag>_<dataset>_s<seed>.csv and results/E3/<tag>_<dataset>_s<seed>.csv"""
import argparse, json, os, sys, time, numpy as np, pandas as pd
from experiment_core import *
from data import MATCHERS, data_file

ALPHAS = {"fing_x_face": [1e-2, 1e-3], "face_x_face": [1e-2, 1e-3, 1e-4], "fing_x_fing": [1e-2, 1e-3, 1e-4]}
METHODS = ["HYP", "MONO", "MONO-rel", "LR-P1-N2", "LR-P1-N2-rel", "LR-P2-N2", "LR-P2-N2-rel", "LR-P1-N3",
           "NLS-N2", "DE-N2", "MAXMONO-K3"]
RULES = ["S1-Marcialis", "S2-Symmetric", "S4-Direct"]
NOISE = [0.05, 0.10]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("seed", type=int)
    ap.add_argument("--frac", type=float, default=0.5); ap.add_argument("--tag", default="main")
    ap.add_argument("--methods", nargs="*", default=METHODS); ap.add_argument("--rules", nargs="*", default=RULES)
    ap.add_argument("--mode", default="corner"); ap.add_argument("--region", nargs=2, type=float, default=None)
    ap.add_argument("--nbins", type=int, default=120); ap.add_argument("--bstep", type=float, default=0.02)
    ap.add_argument("--noise", action="store_true"); ap.add_argument("--alphas", nargs="*", type=float, default=None)
    ap.add_argument("--out", default="../results"); ap.add_argument("--tlim", type=float, default=300)
    a = ap.parse_args()
    B = np.round(np.arange(-3, 1e-9, a.bstep), 4)
    region = tuple(a.region) if a.region else None
    D = dict(np.load(data_file(a.dataset)))
    M = MATCHERS[a.dataset]; alphas = a.alphas or ALPHAS[a.dataset]
    tr, te = make_blocks(D, M, a.seed, a.frac); del D
    G, Gt = tr["_G"], te["_G"]
    os.makedirs(f"{a.out}/E1", exist_ok=True); os.makedirs(f"{a.out}/E3", exist_ok=True)
    rows1, envs, t0 = [], {}, time.time()
    for m in M:
        S, St = tr[m][0], te[m][0]
        gen, imp, gen_t, imp_t = S[G], S[~G], St[Gt], St[~Gt]
        for meth in a.methods:
            env, info = fit_envelope(meth, gen, imp, B=B, mode=a.mode, region=region, nbins=a.nbins, seed=a.seed,
                                     time_limit=a.tlim)
            envs[(meth, m)] = env
            vr, vmax = dominance_on(env, gen_t, imp_t, mode="corner", region=region)
            row = {"dataset": a.dataset, "seed": a.seed, "matcher": m, "method": meth, "tag": a.tag,
                   "terms": json.dumps(env.terms), "kind": env.kind, "xmin": env.xmin, "xmax": env.xmax,
                   "test_viol_rate": vr, "test_viol_max": vmax}
            row.update({k: v for k, v in info.items() if np.isscalar(v)})
            rows1.append(row)
            print(f"[{a.dataset} s{a.seed}] fit {m} {meth} {info['total_time']:.1f}s gap={info.get('gap', np.nan)}", flush=True)
    pd.DataFrame(rows1).to_csv(f"{a.out}/E1/{a.tag}_{a.dataset}_s{a.seed}.csv", index=False)
    rng = np.random.default_rng(1000 + a.seed)
    noisy = {}
    if a.noise:
        for lv in NOISE:
            noisy[lv] = {"_G": Gt}
            for m in M:
                St = te[m][0]; sd = St[~Gt].std()
                noisy[lv][m] = ((St + rng.normal(0, lv * sd, St.shape)).astype(np.float32),)
    rows3 = []
    orders = all_orders(M)
    for alpha in alphas:
        for order in orders:
            for meth in a.methods:
                t1 = time.time()
                r = design_and_evaluate({m: envs[(meth, m)] for m in order}, alpha, tr, te, order)
                r.update({"dataset": a.dataset, "seed": a.seed, "alpha": alpha, "order": ">".join(order),
                          "n_stages": len(order), "method": meth, "kind": "GP", "tag": a.tag, "time": time.time() - t1})
                if a.noise and r["feasible"]:
                    from serial_design import simulate, recover_thresholds
                    d = gp_design([envs[(meth, m)] for m in order], alpha)
                    imp_tr = [np.sort(tr[m][0][~G]) for m in order]; thr = recover_thresholds(d, imp_tr)
                    for lv in NOISE:
                        fa, fr, _, _ = simulate([noisy[lv][m][0] for m in order], Gt, thr)
                        r[f"far_test_n{int(lv*100):02d}"] = fa; r[f"frr_test_n{int(lv*100):02d}"] = fr
                rows3.append(r)
            for rule in a.rules:
                t1 = time.time()
                r = evaluate_rule(rule, alpha, tr, te, order)
                r.update({"dataset": a.dataset, "seed": a.seed, "alpha": alpha, "order": ">".join(order),
                          "n_stages": len(order), "method": rule, "kind": "rule", "tag": a.tag, "time": time.time() - t1})
                rows3.append(r)
        r = evaluate_parallel(alpha, tr, te, M)
        r.update({"dataset": a.dataset, "seed": a.seed, "alpha": alpha, "order": "parallel(" + "+".join(M) + ")",
                  "n_stages": len(M), "method": "P0-Parallel", "kind": "rule", "tag": a.tag, "time": 0.0})
        rows3.append(r)
        print(f"[{a.dataset} s{a.seed}] alpha {alpha} done {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows3).to_csv(f"{a.out}/E3/{a.tag}_{a.dataset}_s{a.seed}.csv", index=False)
    print(f"[{a.dataset} s{a.seed}] finished in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
