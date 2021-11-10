#! /bin/bash
# python3 main.py --classifier_path CNet_t1_ep30_b256.{SSL_None} -m t1 -c CLAM
# python3 main.py --classifier_path CNet_t1_ep30_b256.{SSL_t1_ep250_b512}.finetuned -m t1 -c CLAM

python3 main.py --classifier_path CNet_t2_ep30_b256.{SSL_None} -m t2 -c CLAM
# python3 main.py --classifier_path CNet_t2_ep30_b256.{SSL_t2_ep100_b512}.finetuned -m t2 -c CLAM
