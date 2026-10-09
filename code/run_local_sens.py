"""Launcher for calib_sensitivity_all.py on the 24-core PC (IJIS v08).  D2/D3 jobs need about 3.5 GB each, so at most
HEAVY of them run at once; D1 jobs are light.  Usage: python run_local_sens.py [HEAVY] [LIGHT]"""
import os, subprocess, sys, time
HEAVY = int(sys.argv[1]) if len(sys.argv) > 1 else 6; LIGHT = int(sys.argv[2]) if len(sys.argv) > 2 else 10
jobs = [(ds, s) for ds in ["fing_x_fing", "face_x_face"] for s in range(10)] + [("fing_x_face", s) for s in range(10)]
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
os.makedirs("../results/E3sensall/logs", exist_ok=True)
jobs = [j for j in jobs if not os.path.exists(f"../results/E3sensall/sensall_{j[0]}_s{j[1]}.csv")]
run, t0 = [], time.time()
while jobs or run:
    run = [(p, h) for p, h in run if p.poll() is None]
    for kind, cap in [(True, HEAVY), (False, LIGHT)]:
        while sum(1 for _, h in run if h == kind) < cap:
            nxt = [j for j in jobs if (j[0] != "fing_x_face") == kind]
            if not nxt: break
            j = nxt[0]; jobs.remove(j)
            log = open(f"../results/E3sensall/logs/{j[0]}_s{j[1]}.log", "w")
            run.append((subprocess.Popen([sys.executable, "calib_sensitivity_all.py", j[0], str(j[1])], stdout=log, stderr=subprocess.STDOUT, env=env), kind))
    time.sleep(10)
    print(f"{time.time() - t0:.0f}s running {len(run)} queued {len(jobs)}", flush=True)
open("../results/E3sensall/ALL_DONE", "w").write("done\n")
