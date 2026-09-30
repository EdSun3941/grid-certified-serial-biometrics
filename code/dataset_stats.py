"""Dataset description table: per matcher EER, FRR at FAR targets, genuine/impostor counts, correlations."""
import numpy as np, pandas as pd
from data import roc, MATCHERS, data_file
rows = []; corr = []
for ds in ["fing_x_face", "fing_x_fing", "face_x_face", "lfw_x_fing"]:
    D = np.load(data_file(ds)); G = D["genuine"]
    for m in MATCHERS[ds]:
        S = D["S_" + m]; gen, imp = S[G], S[~G]
        t, far, frr = roc(gen, imp); e = int(np.argmin(np.abs(far - frr)))
        r = {"dataset": ds, "matcher": m, "n_genuine": int(G.sum()), "n_impostor": int((~G).sum()),
             "n_thresholds": len(t) - 1, "EER": float((far[e] + frr[e]) / 2)}
        for a in [1e-2, 1e-3, 1e-4]:
            ok = far <= a; r[f"FRR@FAR<={a:g}"] = float(frr[ok].min())
        rows.append(r)
    Ms = MATCHERS[ds]
    rng = np.random.default_rng(0); imp_idx = np.flatnonzero(~G.ravel()); sub = rng.choice(imp_idx, min(500000, len(imp_idx)), replace=False)
    for i in range(len(Ms)):
        for j in range(i + 1, len(Ms)):
            a, b = D["S_" + Ms[i]], D["S_" + Ms[j]]
            corr.append({"dataset": ds, "pair": f"{Ms[i]}-{Ms[j]}",
                         "genuine_pearson": float(np.corrcoef(a[G], b[G])[0, 1]),
                         "impostor_pearson": float(np.corrcoef(a.ravel()[sub], b.ravel()[sub])[0, 1])})
pd.DataFrame(rows).to_csv("../results/tables/T_datasets.csv", index=False)
pd.DataFrame(corr).to_csv("../results/tables/T_correlations.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string()); print(pd.DataFrame(corr).round(3).to_string())
