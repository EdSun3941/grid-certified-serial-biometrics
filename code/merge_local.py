"""Merge the result files computed on the local 24-core PC (run_local2.py) into results/E3fresh.
A split computed on both machines keeps the cloud file (rule fixed before the comparison); the two are compared by
compare_runs.py and any difference is recorded.  Provenance of every file is written to results/E3fresh_provenance.csv.
Usage: python merge_local.py <folder with the PC's fresh_*.csv files>"""
import glob, os, shutil, sys, pandas as pd
from compare_runs import compare
src = sys.argv[1]; dst = "../results/E3fresh"; prov_f = "../results/E3fresh_provenance.csv"
prov = pd.read_csv(prov_f).set_index("file").to_dict("index") if os.path.exists(prov_f) else {}
for f in sorted(glob.glob(os.path.join(dst, "fresh_*.csv"))):
    prov.setdefault(os.path.basename(f), {"machine": "cloud (2 vCPU)", "checked_against_pc": ""})
for f in sorted(glob.glob(os.path.join(src, "fresh_*.csv"))):
    b = os.path.basename(f); d = os.path.join(dst, b)
    if os.path.exists(d) and prov.get(b, {}).get("machine", "").startswith("cloud"):
        r = compare(d, f)
        if r["ok"]:
            prov[b]["checked_against_pc"] = f"identical rates; max rel. threshold difference {r['max_rel_thr_diff']:.1e}"
        else:                                                    # recorded, cloud file kept (rule fixed before the comparison)
            a_, b_ = [pd.read_csv(x).sort_values(["alpha", "order", "method", "calib"]).reset_index(drop=True) for x in (d, f)]
            rates = [c for c in a_.columns if c.startswith(("far_", "frr_", "stages_"))]
            nd = int(((a_[rates].astype(float) - b_[rates].astype(float)).abs() > 0).any(axis=1).sum())
            prov[b]["checked_against_pc"] = (f"rates differ in {nd} of {len(a_)} designs (largest {r['max_abs_rate_diff']:.4f}); "
                                             f"test FRR/FAR identical in all designs: "
                                             f"{bool((a_[['frr_test', 'far_test']].astype(float) - b_[['frr_test', 'far_test']].astype(float)).abs().max().max() == 0)}; cloud file kept")
    elif not os.path.exists(d):
        shutil.copy2(f, d); prov[b] = {"machine": "local PC (24 cores)", "checked_against_pc": ""}
pd.DataFrame.from_dict(prov, orient="index").rename_axis("file").sort_index().to_csv(prov_f)
p = pd.read_csv(prov_f); print(p.machine.value_counts().to_string()); print(p[p.checked_against_pc.notna()].to_string())
