#! /bin/bash
if [ -n "$1" ]; then
    if [ "$1" == "t1" ]; then
        # python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_None} -m t1 -c ScoreCAM
        python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep250_b512}.finetuned -m t1 -c ScoreCAM
        # python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep250_b512}.fixed -m t1 -c ScoreCAM
        # python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep100_b512.unsupervised}.finetuned -m t1 -c ScoreCAM
    elif [ "$1" == "t2" ]; then
        # python3 main.py --classifier_path CNet_t2_ep30_b256.{Res18_None} -m t2 -c ScoreCAM
        python3 main.py --classifier_path CNet_t2_ep30_b256.{Res18_t2_ep100_b512}.finetuned -m t2 -c ScoreCAM
    elif [ "$1" == "t1ce" ]; then
        python3 main.py --classifier_path CNet_t1ce_ep30_b256.{Res18_t1ce_ep100_b512}.finetuned -m t1ce -c ScoreCAM
    else
    echo "Error modality. Usage: run.sh [modality]"
    fi
else
    echo "Empty modality. Usage: run.sh [modality]"
fi
