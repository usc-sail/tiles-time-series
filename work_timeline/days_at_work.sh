#!/bin/bash

# Extract work day sleep pattern
mainfoldervar=$1
outputfoldervar=$2

echo start to extract feature

# example
# python3.6 days_at_work.py ../../data ../output

python3.6 days_at_work.py -i $mainfoldervar -d $outputfoldervar

