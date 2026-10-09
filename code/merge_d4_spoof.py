"""IJIS v09: D4 spoof analyses evaluated on the machine that computed the D4 thresholds.
The thresholds of D4 splits 2-9 (results/E3fresh) were computed on the 24-core PC.  The SPRT accumulates log-likelihood
ratios that are re-estimated when the designs are evaluated; their last bits differ between the PC and the 2-vCPU Linux
machine, which flips a few test scores lying exactly at a threshold (D4 splits 4 and 7).  spoof_trait.py and
spoof_sens.py were therefore also run for D4 on the PC (python spoof_trait.py lfw_x_fing; python spoof_sens.py
lfw_x_fing).  This script compares those raw files with the ones written here, keeps the local rows of the splits
computed here (0, 1) and the PC rows of splits 2-9, records the comparison, and recomputes the aggregated tables exactly
as spoof_trait.py and spoof_sens.py do.
Usage: python merge_d4_spoof.py <folder with the PC files spoof_trait_raw_lfw_x_fing.csv and spoof_sens_raw_lfw_x_fing.csv>
Output: results/tables/spoof_trait_raw_lfw_x_fing.csv, spoof_sens_raw_lfw_x_fing.csv (merged; the local versions are
kept as *_local.csv), T_rev_spoof_trait.csv, T_spoof_sens.csv, results/D4_spoof_provenance.csv"""
import glob, os, sys, numpy as np, pandas as pd
T = "../results/tables"; PC_SEEDS = set(range(2, 10))


def merge(name, pc_dir, log):
    loc = f"{T}/{name}"; bak = loc.replace(".csv", "_local.csv")
    if not os.path.exists(bak): os.replace(loc, bak)
    c = pd.read_csv(bak, float_precision="round_trip"); p = pd.read_csv(f"{pc_dir}/{name}", float_precision="round_trip")
    num = [x for x in c.columns if x not in ("dataset", "seed", "alpha", "method") and c[x].dtype != object]
    m = c.merge(p, on=["seed", "method"], suffixes=("_l", "_p"))
    for s in sorted(c.seed.unique()):
        mm = m[m.seed == s]
        diff = sorted({r.method for _, r in mm.iterrows() for x in num if not np.isclose(r[f"{x}_l"], r[f"{x}_p"], rtol=0, atol=1e-12, equal_nan=True)})
        log.append(dict(file=name, seed=s, used="PC" if s in PC_SEEDS else "local", methods_differing="|".join(diff)))
    # text-level merge (keeps the exact float representation and the row order of the local file)
    L = open(bak).read().splitlines(); Pl = open(f"{pc_dir}/{name}").read().splitlines()
    assert L[0] == Pl[0], "different columns"
    key = lambda ln: (int(ln.split(",")[1]), ln.split(",")[3])          # columns: dataset, seed, alpha, method, ...
    pcl = {key(ln): ln for ln in Pl[1:]}
    out = [L[0]] + [pcl[key(ln)] if key(ln)[0] in PC_SEEDS else ln for ln in L[1:]]
    open(loc, "w").write("\n".join(out) + "\n")


def main():
    pc_dir = sys.argv[1]; log = []
    merge("spoof_trait_raw_lfw_x_fing.csv", pc_dir, log); merge("spoof_sens_raw_lfw_x_fing.csv", pc_dir, log)
    pd.DataFrame(log).to_csv("../results/D4_spoof_provenance.csv", index=False)
    # aggregation of spoof_trait.py
    A = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in sorted(glob.glob(f"{T}/spoof_trait_raw_*.csv")) if not f.endswith("_local.csv")], ignore_index=True)
    agg = A.groupby(["dataset", "method"]).agg(spoof_trait_max=("spoof_trait_max", "mean"), spoof_trait_mean=("spoof_trait_mean", "mean"),
                                               n=("seed", "size")).reset_index()
    agg["source"] = np.where(agg.dataset == "lfw_x_fing", "results/E3fresh thresholds (calib boot) + test scores; spoof_trait.py (per-trait perfect spoof)",
                             "results/E3b thresholds + test scores; spoof_trait.py (per-trait perfect spoof)")
    agg.to_csv(f"{T}/T_rev_spoof_trait.csv", index=False)
    # aggregation of spoof_sens.py
    A = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in sorted(glob.glob(f"{T}/spoof_sens_raw_*.csv")) if not f.endswith("_local.csv")], ignore_index=True)
    val = [c for c in A.columns if c.endswith("_worst") or c.endswith("_mean") or c in ("frr_test", "frr_test_pad", "acq_gen", "acq_imp")]
    agg = A.groupby(["dataset", "method"])[val].mean().reset_index(); agg["n"] = A.groupby(["dataset", "method"]).size().values
    agg["source"] = "spoof_sens.py: designs of T_rev_selected (E3b) / T_fresh_selected boot (E3fresh, D4) at alpha=1e-3 + test scores"
    agg.to_csv(f"{T}/T_spoof_sens.csv", index=False)
    print(pd.DataFrame(log).to_string())


if __name__ == "__main__":
    main()
