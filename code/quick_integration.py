import numpy as np, time, json
from experiment_core import *
from data import data_file
D = dict(np.load(data_file('fing_x_face')))
M = ['face_C', 'face_G', 'li_V', 'ri_V']
tr, te = make_blocks(D, M, seed=0)
G = tr['_G']
envs = {}; t0 = time.time()
methods = ['HYP', 'MONO', 'LR-P1-N2', 'MONO-rel', 'LR-P1-N2-rel', 'LR-P2-N2-rel', 'NLS-N2-rel', 'DE-N2-rel', 'MAXMONO-K3-rel']
for m in M:
    S = tr[m][0]; gen, imp = S[G], S[~G]
    for meth in methods:
        env, info = fit_envelope(meth, gen, imp, seed=0)
        envs[(meth, m)] = env
        St = te[m][0]; vr, vmax = dominance_on(env, St[te['_G']], St[~te['_G']])
        print(f"{m:7s} {meth:11s} rounds {info['rounds']} nfit {info['n_fit']:4d} ndom {info['n_dom']:5d} sse_dom {info['sse_dom']:.4g} "
              f"maxerr {info['maxerr_dom']:.4f} logarea {info['logarea']:.3f} dom {info['dominates_all']} "
              f"testviol {vr:.3f}/{vmax:.4f} t {info['total_time']:.1f}s gap {info.get('gap', float('nan')):.4f}", flush=True)
print('fit total %.1fs' % (time.time() - t0))
for order in [('ri_V', 'face_C'), ('face_C', 'ri_V')]:
    for alpha in [1e-2, 1e-3]:
        for meth in methods:
            r = design_and_evaluate({m: envs[(meth, m)] for m in order}, alpha, tr, te, order)
            if not r['feasible']: print(order, alpha, meth, 'infeasible'); continue
            print(f"{'>'.join(order):22s} a={alpha:g} {meth:11s} pred_gp {r['frr_pred_gp']:.4f} pred_ex {r['frr_pred_exact']:.4f} "
                  f"emp_model {r['frr_emp_model']:.4f} train {r['frr_train']:.4f}/{r['far_train']:.1e} test {r['frr_test']:.4f}/{r['far_test']:.1e} "
                  f"stages {r['stages_gen']:.2f}/{r['stages_imp']:.2f}")
        for rule in []:
            r = evaluate_rule(rule, alpha, tr, te, order)
            if r['feasible']: print(f"{'>'.join(order):22s} a={alpha:g} {rule:11s} train {r['frr_train']:.4f}/{r['far_train']:.1e} test {r['frr_test']:.4f}/{r['far_test']:.1e} stages {r['stages_gen']:.2f}/{r['stages_imp']:.2f}")
            else: print(order, alpha, rule, 'infeasible/NA')
    print()
r = evaluate_parallel(1e-3, tr, te, M); print('parallel a=1e-3', r)
