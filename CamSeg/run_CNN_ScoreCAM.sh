#! /bin/bash
if [ -n "$1" ]; then
    if [ "$1" == "t1" ]; then
        python3 main.py --classifier_path CNet_t1_ep30_b32.{CNN_t1_ep100_b64}.finetuned -m t1 -c ScoreCAM
    elif [ "$1" == "t2" ]; then
        echo ""
    elif [ "$1" == "t1ce" ]; then
        echo ""
    else
    echo "Error modality. Usage: run.sh [modality]"
    fi
else
    echo "Empty modality. Usage: run.sh [modality]"
fi
