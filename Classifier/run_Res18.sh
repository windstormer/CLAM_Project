#! /bin/bash
python3 main.py --encoder_model SSL_t2_ep100_b512 -m t2 --encoder_mode finetuned -b 256
# python3 main.py --encoder_model SSL_t2_ep250_b512.unsupervised -m t2 --encoder_mode finetuned -b 256
# python3 main.py --encoder_model UNet_t2_ep100_b64 -m t2 --encoder_mode finetuned -b 32
# python3 main.py -m t2 --encoder_model_type UNet -b 64
# python3 main.py --encoder_model DLab_t2_ep95_b64 -m t2 --encoder_mode finetuned -b 32
# python3 main.py -m t2 --encoder_model_type DLab -b 32