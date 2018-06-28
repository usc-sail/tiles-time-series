#!/usr/bin/env bash


# Extract individual timeline
mainfoldervar=$1
outputfoldervar=$2

# sh individual_timeline.sh ../../data ../output
python3.6 individual_timeline.py -i $mainfoldervar -o $outputfoldervar
