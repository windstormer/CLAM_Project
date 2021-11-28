#! /bin/bash
# python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_None} -m t1 -c CLAM
# python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep250_b512}.finetuned -m t1 -c CAM
# python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep250_b512}.finetuned -m t1 -c ScoreCAM
# python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep250_b512}.finetuned -m t1 -c CLAM
# python3 main.py --classifier_path CNet_t1_ep30_b256.{Res18_t1_ep100_b512.unsupervised}.finetuned -m t1 -c CLAM

# python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_None} -m t1 -c CLAM
# python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_t1_ep250_b32}.finetuned -m t1 -c CAM
# python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_t1_ep250_b32}.finetuned -m t1 -c ScoreCAM
# python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_t1_ep250_b32}.finetuned -m t1 -c CLAM
# python3 main.py --classifier_path CNet_t1_ep30_b32.{UNet_t1_ep100_b64.unsupervised}.finetuned -m t1 -c CLAM

python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_None} -m t1 -c CLAM
# python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep65_b32}.finetuned -m t1 -c CAM
# python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep65_b32}.finetuned -m t1 -c ScoreCAM
# python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep65_b32}.finetuned -m t1 -c CLAM
# python3 main.py --classifier_path CNet_t1_ep30_b32.{DLab_t1_ep75_b64.unsupervised}.finetuned -m t1 -c CLAM

# python3 main.py --classifier_path CNet_t1_ep30_b64.{Res50_None} -m t1
# python3 main.py --classifier_path CNet_t1_ep30_b32.{Res50_t1_ep100_b256}.finetuned -m t1 -c CAM
# python3 main.py --classifier_path CNet_t1_ep30_b32.{Res50_t1_ep100_b256}.finetuned -m t1 -c ScoreCAM
# python3 main.py --classifier_path CNet_t1_ep30_b32.{Res50_t1_ep100_b256}.finetuned -m t1 -c CLAM