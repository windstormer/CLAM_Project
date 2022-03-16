#! /bin/bash


#! /bin/bash
if [ -n "$1" ]; then
    if [ "$1" == "t1" ]; then
        # python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_None} -m t1 -c CLAM
        python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep250_b512}.finetuned -m t1 -c CAM
        python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep250_b512}.finetuned -m t1 -c ScoreCAM
        python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep250_b512}.finetuned -m t1 -c CLAM

        # python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep250_b512}.fixed -m t1 -c CLAM
        # python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep100_b512.unsupervised}.finetuned -m t1 -c CLAM

        # python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_None} -m t1 -c CLAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_t1_ep250_b32}.finetuned -m t1 -c CAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_t1_ep250_b32}.finetuned -m t1 -c ScoreCAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_t1_ep250_b32}.finetuned -m t1 -c CLAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_t1_ep250_b32}.fixed -m t1 -c CLAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_t1_ep100_b64.unsupervised}.finetuned -m t1 -c CLAM

        # python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_None} -m t1 -c CLAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep65_b32}.finetuned -m t1 -c CAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep65_b32}.finetuned -m t1 -c ScoreCAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep65_b32}.finetuned -m t1 -c CLAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep65_b32}.fixed -m t1 -c CLAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep75_b64.unsupervised}.finetuned -m t1 -c CLAM

        # python3 main.py --classifier_path CNet_t1_ep30_b32.{CNN_None} -m t1 -c CLAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{CNN_t1_ep100_b64}.finetuned -m t1 -c CAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{CNN_t1_ep100_b64}.finetuned -m t1 -c ScoreCAM
        # python3 main.py --classifier_path CNet_t1_ep30_b32.{CNN_t1_ep100_b64}.finetuned -m t1 -c CLAM
    elif [ "$1" == "t2" ]; then
        # python3 main.py --classifier_path CNet_t2_ep30_b256.{Res18_None} -m t2 -c CAM
        python3 main.py --classifier_path CNet_t2_ep30_b256.{Res18_t2_ep100_b512}.finetuned -m t2 -c CAM
        # python3 main.py --classifier_path CNet_t2_ep30_b256.{Res18_t2_ep100_b512}.finetuned -m t2 -c ScoreCAM
        # python3 main.py --classifier_path CNet_t2_ep30_b256.{Res18_t2_ep100_b512}.finetuned -m t2 -c CLAM

        # python3 main.py --classifier_path CNet_t2_ep30_b32.{UNet_None} -m t2 -c CAM
        # python3 main.py --classifier_path CNet_t2_ep30_b32.{UNet_t2_ep100_b64}.finetuned -m t2 -c CAM
        # python3 main.py --classifier_path CNet_t2_ep30_b32.{UNet_t2_ep100_b64}.finetuned -m t2 -c ScoreCAM
        # python3 main.py --classifier_path CNet_t2_ep30_b32.{UNet_t2_ep100_b64}.finetuned -m t2 -c CLAM

        # python3 main.py --classifier_path CNet_t2_ep30_b32.{DLab_None} -m t2 -c CAM
        # python3 main.py --classifier_path CNet_t2_ep30_b32.{DLab_t2_ep95_b64}.finetuned -m t2 -c CAM
        # python3 main.py --classifier_path CNet_t2_ep30_b32.{DLab_t2_ep95_b64}.finetuned -m t2 -c ScoreCAM
        # python3 main.py --classifier_path CNet_t2_ep30_b32.{DLab_t2_ep95_b64}.finetuned -m t2 -c CLAM

        # python3 main.py --classifier_path CNet_t2_ep30_b32.{Res50_None} -m t2 -c CAM
        # python3 main.py --classifier_path CNet_t2_ep30_b32.{Res50_t2_ep100_b256}.finetuned -m t2 -c CAM

    elif [ "$1" == "t1ce" ]; then
        python3 main.py --classifier_path CNet_t1ce_ep30_b256.{Res18_t1ce_ep100_b512}.finetuned -m t1ce -c CAM
        # python3 main.py --classifier_path CNet_t1ce_ep30_b256.{Res18_t1ce_ep100_b512}.finetuned -m t1ce -c ScoreCAM
        # python3 main.py --classifier_path CNet_t1ce_ep30_b256.{Res18_t1ce_ep100_b512}.finetuned -m t1ce -c CLAM

        # python3 main.py --classifier_path CNet_t1ce_ep30_b32.{UNet_t1ce_ep100_b64}.finetuned -m t1ce -c CAM
        # python3 main.py --classifier_path CNet_t1ce_ep30_b32.{DLab_t1ce_ep50_b64}.finetuned -m t1ce -c CAM
    elif [ "$1" == "flair" ]; then
        python3 main.py --classifier_path CNet_flair_ep30_b256.{Res18_flair_ep100_b512}.finetuned -m flair -c CAM
        # python3 main.py --classifier_path CNet_flair_ep30_b256.{Res18_flair_ep100_b512}.finetuned -m flair -c ScoreCAM
        # python3 main.py --classifier_path CNet_flair_ep30_b256.{Res18_flair_ep100_b512}.finetuned -m flair -c CLAM
    else
    echo "Error modality. Usage: run.sh [modality]"

    fi
else
    echo "Empty modality. Usage: run.sh [modality]"
fi