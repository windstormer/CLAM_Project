#! /bin/bash
# python3 main.py --encoder_model UNet_t1_ep250_b32 -m t1 --encoder_mode finetuned -b 32
# python3 main.py --encoder_model UNet_t1_ep250_b32 -m t1 --encoder_mode fixed -b 32
# python3 main.py --encoder_model UNet_t1_ep100_b64.unsupervised -m t1 --encoder_mode finetuned -b 32
# python3 main.py -m t1 --encoder_model_type UNet -b 32

# python3 main.py --encoder_model UNet_t2_ep100_b64 -m t2 --encoder_mode finetuned -b 32
# python3 main.py --encoder_model UNet_t2_ep100_b64 -m t2 --encoder_mode fixed -b 32

# python3 main.py -m t1ce --encoder_model_type UNet -b 32
python3 main.py --encoder_model UNet_t1ce_ep100_b64 -m t1ce --encoder_mode finetuned -b 32