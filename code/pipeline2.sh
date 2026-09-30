#!/bin/bash
# revision pipeline: wait for the E3b runs, then run the timing jobs on one core with nothing else running
cd "$(dirname "$0")"
while [ ! -f ../results/logs/cal2_DONE ]; do sleep 30; done
./launch.sh jobs_time2.txt 1 time2
echo DONE > ../results/logs/PIPELINE2_DONE
