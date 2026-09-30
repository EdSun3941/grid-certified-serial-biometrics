#!/bin/bash
# round-2 revision pipeline: independent reference -> refit check and E8 (2 procs) -> timing reruns on one core
cd "$(dirname "$0")"
while [ ! -f ../results/logs/ref_DONE ]; do sleep 20; done
./launch.sh jobs_refit.txt 2 refit
./launch.sh jobs_cost.txt 2 cost
./launch.sh jobs_time2.txt 1 time2r
echo DONE > ../results/logs/PIPELINE3_DONE
