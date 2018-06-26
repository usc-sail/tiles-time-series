#!/bin/bash

# Extract work day sleep pattern
mainfoldervar=$1
daysatworkfoldervar=$2

echo start to extract feature

# example
# python3.6 days_at_work.py -i ../../data/keck_wave1/2_preprocessed_data -o ../output/work_timeline

python3.6 days_at_work.py -i $mainfoldervar -d daysatworkfoldervar

