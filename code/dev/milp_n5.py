import sys, numpy as np
sys.path.insert(0, "..")
from scipy.optimize import milp, LinearConstraint, Bounds
from experiment_core import fitting_data, make_blocks, B_DEFAULT
from envelope_solvers import fit_milp_p2, _design, fit_monomial, solve_alpha, AMAX, objective_any
from data import data_file
D = dict(np.load(data_file("fing_x_face"))); tr, te = make_blocks(D, ["face_C"], 0); del D; G = tr["_G"]
S = tr["face_C"][0]; x, y, _, _ = fitting_data(S[G], S[~G]); B = B_DEFAULT
for N in [4, 5]:
    t, v, i = fit_milp_p2(x, y, B, N, time_limit=600)
    print(N, v, i, [(round(a,5), b) for a, b in t])
