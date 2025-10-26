#!/usr/bin/env bash
set -euo pipefail

trap 'echo "Ctrl+C detected, killing all child processes..."; jobs -p | xargs -r kill; exit 1' INT

# Create logs directory if it doesn't exist
mkdir -p logs

# Timestamp for log filenames
timestamp=$(date +"%Y%m%d_%H%M%S")


# /eos/user/d/diptarko/ScoutingAnalyser/scoutingeval/bin/python3 ./Stage0/file_mover.py /eos/cms/store/cmst3/group/daql1scout/run3/ntuples/zb/run392669/L1Scouting/ntuples-L1Scouting-Run2025C-v1-L1SCOUT-392669/250611_122022/0000/ /dev/shm/Data 0& # This loading is to ram
# /eos/user/d/diptarko/ScoutingAnalyser/scoutingeval/bin/python3 ./Stage0/file_mover.py /eos/cms/store/cmst3/group/daql1scout/run3/ntuples/zb/run392997/L1Scouting/ntuples-L1Scouting-Run2025C-v1-L1SCOUT-392997/250611_130406/0000/ /dev/shm/Data 0&
# /eos/user/d/diptarko/ScoutingAnalyser/scoutingeval/bin/python3 ./Stage0/file_mover.py /eos/cms/store/cmst3/group/daql1scout/run3/ntuples/zb/run393111/L1Scouting/ntuples-L1Scouting-Run2025C-v1-L1SCOUT-393111/250611_131938/0000/ /dev/shm/Data 0&
# /eos/user/d/diptarko/ScoutingAnalyser/scoutingeval/bin/python3 ./Stage0/file_mover.py ./Data/ /dev/shm/Data 1& # This will be later removed by CMSSW part

bash ./Stage0/wrap.sh >> logs/stage0_${timestamp}.log 2>&1 &
bash ./Stage1/wrap.sh >> logs/stage1_${timestamp}.log 2>&1 &
bash ./Stage2/wrap.sh >> logs/stage2_${timestamp}.log 2>&1 &
bash ./Stage3/wrap.sh >> logs/stage3_${timestamp}.log 2>&1 &
bash ./Stage4/wrap.sh >> logs/stage4_${timestamp}.log 2>&1 &

stdbuf -oL -eL bash ./Tracker/wrap.sh >> logs/tracker_${timestamp}.log 2>&1 &

wait