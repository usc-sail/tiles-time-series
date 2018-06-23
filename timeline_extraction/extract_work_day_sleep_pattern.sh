#!/bin/bash

# Extract work day sleep pattern
mainfoldervar=$1
outputfoldervar=$2
typevar=$3

echo start to extract feature

# 1. Extract work schedule
python3.6 extract_work_schedule.py -i $mainfoldervar -o $outputfoldervar -v 6

# 2. Extract sleep time line
python3.6 extract_sleep_timeline.py -t combined -i $mainfoldervar -o $outputfoldervar

# 3. Extract work day sleep pattern
python3.6 extract_work_day_sleep_pattern.py -t combined -i $mainfoldervar -o $outputfoldervar

