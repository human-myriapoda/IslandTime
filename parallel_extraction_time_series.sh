#!/bin/bash

source C:/Users/myriampe/anaconda3/etc/profile.d/conda.sh
# source C:/Users/mp222/AppData/Local/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate IslandTimeEnv
#echo "Current Conda environment: $CONDA_PREFIX"

# Set the number of times to run the script
num_runs=15

# Set the Python script you want to run
python_script='./parallel_extraction_time_series.py'

for ((i = 1; i <= num_runs; i++)); do
    echo "Running the script for the $i time"
    if python "$python_script"; then
        echo "Script executed successfully."
    else
        echo "Script crashed or returned an error, moving to the next iteration."
        continue
    fi
done