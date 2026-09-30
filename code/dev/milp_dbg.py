import sys, numpy as np
sys.path.insert(0, "..")
from scipy.optimize import milp, LinearConstraint, Bounds
from experiment_core import fitting_data, make_blocks, B_DEFAULT
from envelope_solvers import _design, fit_monomial, AMAX, solve_alpha
from data import data_file
D = dict(np.load(data_file("fing_x_face"))); tr, te = make_blocks(D, ["face_C"], 0); del D; G = tr["_G"]
S = tr["face_C"][0]; x, y, _, _ = fitting_data(S[G], S[~G]); B = B_DEFAULT; N = 5
Phi, yv = _design(x, y, B, False); n, m = Phi.shape
mono, U = fit_monomial(x, y, B, "P2")
acap = np.minimum(AMAX, np.min((yv + U)[:, None] / Phi, axis=0)); Ps = Phi * acap[None, :]
c = np.r_[np.zeros(2 * m), 1.0]
cons = [LinearConstraint(np.hstack([Ps, np.zeros((n, m)), -np.ones((n, 1))]), -np.inf, yv),
        LinearConstraint(np.hstack([Ps, np.zeros((n, m)), np.zeros((n, 1))]), yv, np.inf),
        LinearConstraint(np.hstack([np.eye(m), -np.eye(m), np.zeros((m, 1))]), -np.inf, 0.0),
        LinearConstraint(np.r_[np.zeros(m), np.ones(m), 0.0][None, :], -np.inf, N)]
res = milp(c, constraints=cons, integrality=np.r_[np.zeros(m), np.ones(m), 0], bounds=Bounds(np.zeros(2*m+1), np.r_[np.ones(m), np.ones(m), U]),
           options={"time_limit": 600, "mip_rel_gap": 1e-4})
a = res.x[:m]; z = res.x[m:2*m]; print("sigma", res.x[-1], "status", res.status)
for b in np.where((a > 1e-9) | (z > 1e-9))[0]: print(B[b], "a'", a[b], "z", z[b])
g = Ps @ a; print("max over", (g - yv).max(), "min under", (g - yv).min())
sel = np.where(z > 0.5)[0]; v, al = solve_alpha(Phi[:, sel], yv, "P2"); print("resolve", v)
v2, al2 = solve_alpha(Ps[:, sel], yv, "P2"); print("resolve scaled", v2)
