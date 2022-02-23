#! /bin/bash
if [ -n "$1" ]; then
    if [ "$1" == "t1" ]; then
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_None} -m t1 -c CLAM 
        python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep65_b32}.finetuned -m t1 -c CLAM 
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep65_b32}.fixed -m t1 -c CLAM 
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep75_b64.unsupervised}.finetuned -m t1 -c CLAM
    elif [ "$1" == "t2" ]; then
        # python3 main.py --classifier_path CNet_t2_ep30_b32.{DLab_None} -m t2 -c CLAM 
        python3 main.py --classifier_path CNet_t2_ep30_b32.{DLab_t2_ep95_b64}.finetuned -m t2 -c CLAM
    elif [ "$1" == "t1ce" ]; then
        python3 main.py --classifier_path CNet_t1ce_ep30_b32.{DLab_t1ce_ep50_b64}.finetuned -m t1ce -c CLAM
    else
    echo "Error modality. Usage: run.sh [modality]"
    fi
else
    echo "Empty modality. Usage: run.sh [modality]"
fi
