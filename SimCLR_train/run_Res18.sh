#! /bin/bash
if [ -n "$1" ]; then
    if [ "$1" == "t1" ]; then
        python3 main.py -b 512 -e 100 -m t1
    elif [ "$1" == "t2" ]; then
        python3 main.py -b 512 -e 100 -m t2
    elif [ "$1" == "t1ce" ]; then
        python3 main.py -b 512 -e 100 -m t1ce
    elif [ "$1" == "flair" ]; then
        python3 main.py -b 512 -e 100 -m flair
    else
    echo "Error modality. Usage: run.sh [modality]"
    fi
else
    echo "Empty modality. Usage: run.sh [modality]"
fi