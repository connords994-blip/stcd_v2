#!/usr/bin/env bash
# Run the DVSNOISE20 B3 no-regression bake-off one recording per subprocess so
# peak RAM stays bounded against the 2.5 GB .mat loads on the 6 GB WSL cap.
set -u
cd ~/stcd
source .venv/bin/activate
SCENES="bike classroom conference soccer stairs toys"
LOG=/tmp/b3_dvs.log
: > "$LOG"
for f in data/dvsnoise20/2_mat/*.mat; do
  base=$(basename "$f"); name="${base%%-*}"
  for s in $SCENES; do
    if [ "$name" = "$s" ]; then
      PYTHONPATH=src python scripts/bakeoff_dvsnoise.py "$f" >> "$LOG" 2>>/tmp/b3_dvs.err
    fi
  done
done
echo "DONE" >> "$LOG"
