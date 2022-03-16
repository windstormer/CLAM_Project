#! /bin/bash
python3 main.py --encoder_model CNN_t1_ep100_b64 -m t1 --encoder_mode finetuned -b 32
# python3 main.py --encoder_model CNN_t1_ep100_b64 -m t1 --encoder_mode fixed -b 32
# python3 main.py --encoder_model CNN_t1_ep100_b64.unsupervised -m t1 --encoder_mode finetuned -b 32
# python3 main.py -m t1 --encoder_model_type CNN -b 32

# python3 main.py --encoder_model CNN_t2_ep100_b64 -m t2 --encoder_mode finetuned -b 32
# python3 main.py --encoder_model CNN_t2_ep100_b64 -m t2 --encoder_mode fixed -b 32