"""Round-2 diagnostic of the subject-bootstrap calibration (reviewers 1 and 3).
(a) For the selected designs of every method (T_rev_selected.csv): mean training FAR, mean test FAR, and the mean and
    split-to-split SD of the shift (test - train), all divided by alpha.
(b) For the selected proposed designs (LR-P1-N2): the bootstrap SD of the training FAR at the deployed thresholds,
    recomputed with the same subject weights as in calibration (seed 1000 + split), divided by alpha.
Output: results/tables/T_rev_calib_diag.csv"""
import json, numpy as np, pandas as pd
import serial_design as sd
from serial_design import _boot_weights
from experiment_core import make_blocks
from data import MATCHERS, data_file, split_subjects

sel = pd.read_csv("../results/tables/T_rev_selected.csv")
METHODS = ["LR-P1-N2", "S1-Marcialis", "S4-Direct", "S3-SPRT", "P0-Parallel", "P1-LLR", "P2-LogReg"]
rows = []
for (ds, a, m), g in sel[sel.method.isin(METHODS)].groupby(["dataset", "alpha", "method"]):
    sh = (g.far_test - g.far_train) / a
    rows.append(dict(dataset=ds, alpha=a, method=m, n=len(g), far_train_over_alpha=float((g.far_train / a).mean()),
                     far_test_over_alpha=float((g.far_test / a).mean()), shift_mean=float(sh.mean()), shift_sd=float(sh.std(ddof=1)),
                     shift_sd_over_sqrt2=float(sh.std(ddof=1) / np.sqrt(2))))
diag = pd.DataFrame(rows)
# (b) bootstrap SD at the deployed thresholds of the selected proposed designs
bs = []
for ds in ["fing_x_fing", "face_x_face", "fing_x_face"]:
    D = dict(np.load(data_file(ds))); M = MATCHERS[ds]
    for seed in range(10):
        e3 = pd.read_csv(f"../results/E3b/main_{ds}_s{seed}.csv"); e3 = e3[(e3.method == "LR-P1-N2") & (e3.feasible == True)]
        r_tr, c_tr, _, _ = split_subjects(D["row_subject"], D["col_subject"], seed)
        sd.BOOT.update(row_subj=D["row_subject"][r_tr], col_subj=D["col_subject"][c_tr], B=300, seed=1000 + seed)
        tr, _ = make_blocks(D, M, seed); G = tr["_G"]
        wr, wc = _boot_weights(300, 1000 + seed); gi, gj = np.nonzero(G)
        den = wr.sum(1) * wc.sum(1) - (wr[:, gi] * wc[:, gj]).sum(1)
        for a in sorted(e3.alpha.unique()):
            s = sel[(sel.dataset == ds) & (sel.seed == seed) & np.isclose(sel.alpha, a) & (sel.method == "LR-P1-N2")]
            if len(s) == 0: continue
            sds, p95s = [], []
            for order in s.orders.iloc[0].split("|"):
                r = e3[np.isclose(e3.alpha, a) & (e3.order == order)].iloc[0]; thr = json.loads(r.thr); chain = order.split(">")
                decided = np.zeros(G.shape, bool); acc = np.zeros(G.shape, bool)
                for k, m in enumerate(chain):
                    und = ~decided; sc = tr[m][0]
                    if k < len(chain) - 1:
                        ta, trj = thr[k]; x = und & (sc >= ta); y = und & (sc < trj); acc |= x; decided |= x | y
                    else:
                        acc |= und & (sc >= thr[k])
                ai, aj = np.nonzero(acc & ~G); far_b = (wr[:, ai] * wc[:, aj]).sum(1) / den
                sds.append(far_b.std(ddof=1) / a); p95s.append(np.quantile(far_b, 0.95) / a)
            bs.append(dict(dataset=ds, alpha=a, seed=seed, boot_sd_over_alpha=float(np.mean(sds)), boot_p95_over_alpha=float(np.mean(p95s))))
        print(ds, seed, flush=True)
    del D
bs = pd.DataFrame(bs); bs.to_csv("../results/tables/T_rev_calib_boot.csv", index=False)
agg = bs.groupby(["dataset", "alpha"]).agg(boot_sd_min=("boot_sd_over_alpha", "min"), boot_sd_max=("boot_sd_over_alpha", "max"),
                                           boot_sd_mean=("boot_sd_over_alpha", "mean"), boot_p95_mean=("boot_p95_over_alpha", "mean")).reset_index()
agg["method"] = "LR-P1-N2"
diag = diag.merge(agg, on=["dataset", "alpha", "method"], how="left")
diag["source"] = "T_rev_selected.csv; results/E3b (thresholds); calib_diag.py"
diag.to_csv("../results/tables/T_rev_calib_diag.csv", index=False)
print(diag.drop(columns="source").round(3).to_string())
