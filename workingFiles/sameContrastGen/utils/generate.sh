#!/bin/bash

# Step 1 args
GEN_DATA_DIR="/storage/data/test" # Path to generation dataset, must be a directory with images and labels subfolder
K_MIN=1 # Min clustering arg
K_MAX=3 # Max clustering arg
NUM_GEN=1 # Number of clustered images to generate for each
SYNTH_DATA_OUT="./test_synth_data"

source /home/ubuntu/miniconda3/bin/activate
# Step 1 runs first, create a local directory to store step1 files, cleaned later by runner
# python3 step1.py --data_dir "/home/ary/synthetic-generalisation/fatdata" --k_min 5 --k_max 10 --num_gen 1
conda activate step1
python3 step1.py --data_dir $GEN_DATA_DIR --k_min $K_MIN --k_max $K_MAX --num_gen $NUM_GEN


# Step 2
python3 step2.py


# Step 3
conda activate synthseg_38
python step3.py --data_dir $GEN_DATA_DIR

# Step 4
python3 step4.py --data_dir $GEN_DATA_DIR --out_dir $SYNTH_DATA_OUT

rm -r step1data
rm -r step2data
rm -r step3data