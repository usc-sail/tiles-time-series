#!/usr/bin/env bash

mainfoldervar=$1
outputfoldervar=$2

# 1. extract sleep timeline
# 2. extract days_at_work timeline
# 3. extract signal recording start and end timeline
# 4. create individual based timeline

# Example command line:
# sh run.sh ../../data ../output

# extract sleep timeline
echo start to extract sleep_timeline
cd sleep_timeline

python3.6 extract_sleep_timeline.py -t sleep_summary -i $mainfoldervar -o $outputfoldervar

cd ..

# Extract work day sleep pattern
cd work_timeline
echo start to days at work

python3.6 days_at_work.py -i $mainfoldervar -o $outputfoldervar

# Extract work day sleep pattern
python3.6 signal_recording_start_end.py -i $mainfoldervar -o $outputfoldervar

cd ..
# Extract work day sleep pattern
cd individual_timeline
echo start to days at work
