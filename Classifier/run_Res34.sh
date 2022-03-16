#! /bin/bash
if [ -n "$1" ]; then
    if [ "$1" == "t1" ]; then
        python3 main.py --encoder_model Res34_t1_ep100_b256 -m t1 --encoder_mode finetuned -b 256
    elif [ "$1" == "t2" ]; then
        python3 main.py --encoder_model Res34_t2_ep100_b256 -m t2 --encoder_mode finetuned -b 256
    elif [ "$1" == "t1ce" ]; then
        python3 main.py --encoder_model Res34_t1ce_ep100_b256 -m t1ce --encoder_mode finetuned -b 256
    elif [ "$1" == "flair" ]; then
        python3 main.py --encoder_model Res34_flair_ep100_b256 -m flair --encoder_mode finetuned -b 256
    else
    echo "Error modality. Usage: run.sh [modality]"
    fi
else
    echo "Empty modality. Usage: run.sh [modality]"
fi
