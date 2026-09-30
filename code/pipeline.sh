#!/bin/bash
cd "$(dirname "$0")"
./launch.sh jobs_cal.txt 2 cal
./launch.sh jobs_abl.txt 2 abl
./launch.sh jobs_time.txt 1 time
echo PIPELINE_DONE > ../results/logs/PIPELINE_DONE
