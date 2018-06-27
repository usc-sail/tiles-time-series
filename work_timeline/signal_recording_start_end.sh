#!/usr/bin/env bash

# Extract work day sleep pattern
mainfoldervar=$1
daysatworkfoldervar=$2
recordingtimelinefoldervar=$3

echo start to extract feature

# example
# sh signal_recording_start_end.sh ../../data/keck_wave1/2_preprocessed_data ../output/days_at_work ../output/recording_timeline
# python3.6 signal_recording_start_end.py -i ../../data/keck_wave1/2_preprocessed_data -o ../output/work_timeline -r ../output/recording_timeline

python3.6 signal_recording_start_end.py -i $mainfoldervar -d $daysatworkfoldervar -r $recordingtimelinefoldervar
