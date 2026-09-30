import sys, time, numpy as np
sys.path.insert(0, "..")
import serial_design as sd
from data import data_file, MATCHERS, split_subjects
from experiment_core import make_blocks
for ds in ["fing_x_face", "face_x_face", "fing_x_fing"]:
    D = dict(np.load(data_file(ds))); M = MATCHERS[ds]
    r_tr, c_tr, _, _ = split_subjects(D["row_subject"], D["col_subject"], 0)
    sd.BOOT.update(row_subj=D["row_subject"][r_tr], col_subj=D["col_subject"][c_tr], B=300, seed=0)
    tr, te = make_blocks(D, M, 0); del D; G = tr["_G"]
    for m in M[:1]:
        for a in [1e-2, 1e-3, 1e-4]:
            t = time.time(); tb = sd.final_threshold_boot([tr[m][0]], G, [], a); tb_t = time.time() - t
            t = time.time(); tc = sd.final_threshold_conf([tr[m][0]], G, [], a, 0.95); tc_t = time.time() - t
            tp = sd.final_threshold_conf([tr[m][0]], G, [], a, None)
            S = tr[m][0]; I = ~G
            far = lambda th: float((S[I] >= th).mean()) if th is not None else float('nan')
            print(f"{ds} {m} a={a}: point {far(tp):.2e}  CP {far(tc):.2e} ({tc_t:.1f}s)  boot {far(tb):.2e} ({tb_t:.1f}s)", flush=True)
