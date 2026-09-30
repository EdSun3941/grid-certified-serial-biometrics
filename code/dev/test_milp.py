import sys, time, numpy as np
sys.path.insert(0, "..")
from experiment_core import fitting_data, make_blocks, B_DEFAULT
from envelope_solvers import fit_milp_p2, fit_exact_dual, fit_lr
from data import data_file
for ds, mm in [("fing_x_face", "face_C"), ("fing_x_fing", "li_V")]:
    D = dict(np.load(data_file(ds))); tr, te = make_blocks(D, [mm], 0); del D; G = tr["_G"]
    S = tr[mm][0]; x, y, _, _ = fitting_data(S[G], S[~G])
    for N in [1, 2, 3, 4]:
        t, v, i = fit_milp_p2(x, y, B_DEFAULT, N, time_limit=120)
        t2, v2, i2 = fit_exact_dual(x, y, B_DEFAULT, N, "P2", time_limit=120)
        print(ds, mm, "N", N, f"MILP {v:.6g} {i['time']:.2f}s status {i['status']} | exactLRBB {v2:.6g} gap {i2['gap']:.4f} rootgap {i2['root_gap']:.3f} nodes {i2['nodes']} {i2['time']:.2f}s", flush=True)
