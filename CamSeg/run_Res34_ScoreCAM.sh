#! /bin/bash
if [ -n "$1" ]; then
    if [ "$1" == "t1" ]; then
        python3 main.py --classifier_path CNet_t1_ep30_b256.{Res34_t1_ep100_b256}.finetuned -m t1 -c ScoreCAM
    elif [ "$1" == "t2" ]; then
        python3 main.py --classifier_path CNet_t2_ep30_b256.{Res34_t2_ep100_b256}.finetuned -m t2 -c ScoreCAM
    elif [ "$1" == "t1ce" ]; then
        python3 main.py --classifier_path CNet_t1ce_ep30_b256.{Res34_t1ce_ep100_b256}.finetuned -m t1ce -c ScoreCAM
    elif [ "$1" == "flair" ]; then
        python3 main.py --classifier_path CNet_flair_ep30_b256.{Res34_flair_ep100_b256}.finetuned -m flair -c ScoreCAM
    else
    echo "Error modality. Usage: run.sh [modality]"
    fi
else
    echo "Empty modality. Usage: run.sh [modality]"
fi
