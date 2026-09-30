#!/bin/bash
# round-2: wait for E8, then update E3b LR rows with exact-dual envelopes (2 procs), then timing reruns alone on one core
cd "$(dirname "$0")"
while [ ! -f ../results/logs/cost_DONE ]; do sleep 20; done
./launch.sh jobs_cal2xd.txt 2 cal2xd
./launch.sh jobs_time2.txt 1 time2r
echo DONE > ../results/logs/PIPELINE4_DONE
