import numpy as np, time, sys
from data import *
from envelope_solvers import *
d = dict(np.load('../data/bssr1_fing_x_face_old.npz'))
d['row_subject'], d['col_subject'] = d['users'], d['enrollees']
rtr, ctr, rte, cte = split_subjects(d['row_subject'], d['col_subject'], seed=0)
G = d['genuine'][np.ix_(rtr, ctr)]
B = np.round(np.arange(-3, 1e-9, 0.02), 2)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
for m in ['S_face_C', 'S_ri_V']:
    S = d[m][np.ix_(rtr, ctr)]
    xf, yf, xd, yd = fitting_sets(S[G], S[~G])
    print(f'== {m}: fit pts {len(xf)}, dominance pts {len(xd)}, x range [{xf.min():.2e},{xf.max():.2f}]')
    for obj in ['P1', 'P2']:
        t = time.time(); te, ve, ie = fit_enum(xf, yf, B, N, obj); 
        print(f'  {obj} exact enum N={N}: {ve:.6g}  terms {[(round(a,6),b) for a,b in te]}  {ie}')
        for mode in ['root', 'bb']:
            for ob in [False, True]:
                tl, vl, il = fit_lr(xf, yf, B, N, obj, mode=mode, obbt=ob, time_limit=120)
                ok = il['lb'] <= ve * (1 + 1e-9) + 1e-12
                print(f'  {obj} LR {mode:4s} obbt={ob!s:5}: UB {vl:.6g}  LB {il["lb"]:.6g}  gap {100*il["gap"]:.3f}%  '
                      f'vs exact {100*(vl-ve)/ve:+.4f}%  LB_valid={ok}  nodes {il["nodes"]} qp {il["n_qp"]} time {il["time"]:.1f}s')
