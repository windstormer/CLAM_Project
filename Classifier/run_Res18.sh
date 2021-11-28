#! /bin/bash
# python3 main.py --encoder_model Res18_t1_ep250_b512 -m t1 --encoder_mode finetuned -b 256
# python3 main.py --encoder_model Res18_t1_ep100_b512.unsupervised -m t1 --encoder_mode finetuned -b 256
# python3 main.py -m t1 --encoder_model_type SSL -b 128
python3 main.py --encoder_model Res18_t1_ep250_b512 -m t1 --encoder_mode fixed -b 256