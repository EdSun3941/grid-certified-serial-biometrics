import sys, time, numpy as np
sys.path.insert(0, "..")
from experiment_core import fitting_data, make_blocks, B_DEFAULT
from envelope_solvers import fit_exact_dual, fit_enum
from data import data_file
for ds, mm in [("fing_x_face", "face_C"), ("fing_x_fing", "li_V"), ("face_x_face", "face_G")]:
    D = dict(np.load(data_file(ds))); tr, te = make_blocks(D, [mm], 0); del D; G = tr["_G"]
    S = tr[mm][0]; x, y, _, _ = fitting_data(S[G], S[~G])
    for obj in ["P1", "P2"]:
        for N in [1, 2, 3, 5]:
            t = time.time(); terms, U, info = fit_exact_dual(x, y, B_DEFAULT, N, obj, time_limit=300)
            ref = ""
            if N <= 2:
                _, Ue, _ = fit_enum(x, y, B_DEFAULT, N, obj); ref = f" enum={Ue:.6g}"
            print(f"{ds} {mm} {obj} N={N}: U={U:.6g}{ref} gap={100*info['gap']:.4f}% rootgap={100*info['root_gap']:.4f}% nodes={info['nodes']} t={info['time']:.2f}s terms={[(round(a,4),b) for a,b in terms]}", flush=True)
