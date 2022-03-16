#! /bin/bash
# python3 main.py --encoder_model Res18_t1_ep250_b512 -m t1 --encoder_mode finetuned -b 256
# python3 main.py --encoder_model Res18_t1_ep250_b512 -m t1 --encoder_mode fixed -b 256
# python3 main.py --encoder_model Res18_t1_ep100_b512.unsupervised -m t1 --encoder_mode finetuned -b 256
# python3 main.py -m t1 --encoder_model_type Res18 -b 128

# python3 main.py --encoder_model Res18_t2_ep100_b512 -m t2 --encoder_mode finetuned -b 256
# python3 main.py --encoder_model Res18_t2_ep100_b512 -m t2 --encoder_mode fixed -b 256

# python3 main.py -m t1ce --encoder_model_type Res18 -b 128
# python3 main.py --encoder_model Res18_t1ce_ep100_b512 -m t1ce --encoder_mode finetuned -b 256
python3 main.py --encoder_model Res18_flair_ep100_b512 -m flair --encoder_mode finetuned -b 256