#! /bin/bash
python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_None} -m t1 -c ScoreCAM
python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep65_b32}.finetuned -m t1 -c ScoreCAM

# python3 main.py --classifier_path CNet_t2_ep30_b32.{DLab_None} -m t2 -c ScoreCAM
# python3 main.py --classifier_path CNet_t2_ep30_b32.{DLab_t2_ep95_b64}.finetuned -m t2 -c ScoreCAM