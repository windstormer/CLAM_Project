#! /bin/bash

module purge
module load miniconda3

source activate torch

#Operations
dir=$(dirname "$1")
cd $dir

script_name=$(basename "$1")
# echo $script_name $2
srun $script_name $2
# for script in "$@"
# do
#     script_name=$(basename "$script")
#     srun $script_name
# done