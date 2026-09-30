"""Parallel launcher for the fresh-seed replication (D1-D3, splits 10-29) and subset D4 (splits 0-9) on a multi-core PC.
Each job runs `python run_fresh.py <dataset> <seed>` in its own single-threaded process (BLAS threads = 1), so the
results do not depend on the number of workers.  A job whose output already exists is skipped.
Memory-aware: a job starts only if the sum of the nominal peak memory of the running jobs stays within the budget.
Usage: python run_local.py [workers=14] [memory budget in GB=11]
Logs: ../results/logs_local/<dataset>_s<seed>.log; progress: ../results/logs_local/progress.csv; done flag: DONE."""
import os, subprocess, sys, time

W = int(sys.argv[1]) if len(sys.argv) > 1 else 14
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 11.0
MEM = {"fing_x_fing": 2.0, "face_x_face": 1.2, "lfw_x_fing": 0.6, "fing_x_face": 0.6}   # nominal peak GB per job
JOBS = ([("fing_x_fing", s) for s in range(10, 30)] + [("face_x_face", s) for s in range(10, 30)]
        + [("lfw_x_fing", s) for s in range(10)] + [("fing_x_face", s) for s in range(10, 30)])   # longest first
LOG = "../results/logs_local"


def out(ds, s):
    return f"../results/E3fresh/fresh_{ds}_s{s}.csv"


def main():
    os.makedirs(LOG, exist_ok=True); os.makedirs("../results/E3fresh", exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    pending = [j for j in JOBS if not os.path.exists(out(*j))]; running = {}
    prog = open(os.path.join(LOG, "progress.csv"), "a", buffering=1)
    prog.write(f"# start {time.strftime('%Y-%m-%d %H:%M:%S')} workers={W} budget={BUDGET} pending={len(pending)}\n")
    while pending or running:
        for j, (p, t0, fh) in list(running.items()):
            if p.poll() is not None:
                fh.close(); del running[j]
                prog.write(f"{j[0]},{j[1]},{p.returncode},{time.time() - t0:.0f},{os.path.exists(out(*j))}\n")
        used = sum(MEM[j[0]] for j in running)
        for j in list(pending):
            if len(running) >= W: break
            if used + MEM[j[0]] > BUDGET: continue
            fh = open(os.path.join(LOG, f"{j[0]}_s{j[1]}.log"), "w")
            p = subprocess.Popen([sys.executable, "run_fresh.py", j[0], str(j[1])], stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[j] = (p, time.time(), fh); pending.remove(j); used += MEM[j[0]]
        time.sleep(5)
    prog.write(f"# end {time.strftime('%Y-%m-%d %H:%M:%S')}\n"); prog.close()
    open(os.path.join(LOG, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
