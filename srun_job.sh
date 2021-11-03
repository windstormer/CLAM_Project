#! /bin/bash

module purge
module load miniconda3

source activate torch

#Operations
dir=$(dirname "$1")
cd $dir

for script in "$@"
do
    script_name=$(basename "$script")
    srun $script_name
done