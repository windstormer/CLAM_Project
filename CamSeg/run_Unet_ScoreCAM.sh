#! /bin/bash
python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_None} -m t1 -c ScoreCAM
python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_t1_ep250_b32}.finetuned -m t1 -c ScoreCAM

# python3 main.py --classifier_path CNet_t2_ep30_b32.{UNet_None} -m t2 -c ScoreCAM
# python3 main.py --classifier_path CNet_t2_ep30_b32.{UNet_t2_ep100_b64}.finetuned -m t2 -c ScoreCAM

