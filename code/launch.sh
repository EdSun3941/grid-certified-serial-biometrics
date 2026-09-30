#!/bin/bash
# usage: launch.sh <jobfile> <nproc> <logprefix>
JOBS=$1; NP=$2; PFX=$3
mkdir -p ../results/logs
i=0
while read -r cmd; do
  [ -z "$cmd" ] && continue
  i=$((i+1))
  while [ "$(jobs -rp | wc -l)" -ge "$NP" ]; do sleep 2; done
  ( eval "$cmd" > "../results/logs/${PFX}_$(printf %03d $i).log" 2>&1 ) &
done < "$JOBS"
wait
echo ALL_DONE > "../results/logs/${PFX}_DONE"
