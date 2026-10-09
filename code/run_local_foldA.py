"""Launcher for run_foldA.py on the 24-core PC (IJIS v09): the D2 and D3 splits of the held-out (fresh) replication
(seeds 10-29); D1 and D4 ran in the cloud.  Usage: python run_local_foldA.py [NPROC]"""
import os, subprocess, sys, time
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
jobs = [(ds, s) for ds in ["fing_x_fing", "face_x_face"] for s in range(10, 30)]
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
os.makedirs("../results/E3foldA/logs", exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f"../results/E3foldA/foldA_{j[0]}_s{j[1]}.csv")]
run, t0 = [], time.time()
while jobs or run:
    run = [p for p in run if p.poll() is None]
    while jobs and len(run) < NP:
        j = jobs.pop(0); log = open(f"../results/E3foldA/logs/{j[0]}_s{j[1]}.log", "w")
        run.append(subprocess.Popen([sys.executable, "run_foldA.py", j[0], str(j[1])], stdout=log, stderr=subprocess.STDOUT, env=env))
    time.sleep(10)
    print(f"{time.time() - t0:.0f}s running {len(run)} queued {len(jobs)}", flush=True)
open("../results/E3foldA/ALL_DONE", "w").write("done\n")
