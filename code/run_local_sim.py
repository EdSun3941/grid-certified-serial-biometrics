"""Launcher for sim_controlled.py on a multi-core PC (IJIS v08): runs the job list with at most W concurrent
single-threaded processes.  Usage: python run_local_sim.py [W]"""
import os, subprocess, sys, time
W = int(sys.argv[1]) if len(sys.argv) > 1 else 24
JOBS = []
for N, R, chunk in [(500, 200, 10), (1500, 100, 5)]:
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8]:
        for r0 in range(0, R, chunk):
            JOBS.append((N, rho, r0, r0 + chunk))
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
os.makedirs("../results/E9sim/logs", exist_ok=True)
JOBS = [j for j in JOBS if not os.path.exists(f"../results/E9sim/sim_N{j[0]}_r{j[1]:.1f}_{j[2]:03d}_{j[3]:03d}.csv")]
JOBS.sort(key=lambda j: -j[0])                      # long jobs first
run, t0 = [], time.time()
while JOBS or run:
    run = [p for p in run if p.poll() is None]
    while JOBS and len(run) < W:
        N, rho, r0, r1 = JOBS.pop(0)
        log = open(f"../results/E9sim/logs/N{N}_r{rho:.1f}_{r0:03d}.log", "w")
        run.append(subprocess.Popen([sys.executable, "sim_controlled.py", str(N), str(rho), str(r0), str(r1)], stdout=log, stderr=subprocess.STDOUT, env=env))
    time.sleep(5)
    print(f"{time.time() - t0:.0f}s running {len(run)} queued {len(JOBS)}", flush=True)
open("../results/E9sim/ALL_DONE", "w").write("done\n")
