! /bin/bash
# python3 main.py --encoder_model SSL_t1_ep250_b512 -m t1 --encoder_mode finetuned -b 256
# python3 main.py --encoder_model SSL_t1ce_ep250_b512 -m t1ce --encoder_mode finetuned -b 256
# python3 main.py --encoder_model SSL_t1_ep250_b512.unsupervised -m t1 --encoder_mode finetuned -b 256
# python3 main.py --encoder_model UNet_t1_ep250_b32.relu -m t1 --encoder_mode finetuned -b 32
# python3 main.py -m t1 --encoder_model_type UNet -b 64
# python3 main.py --encoder_model DLab_t1_ep65_b32.relu -m t1 --encoder_mode finetuned -b 32 -e 15
# python3 main.py -m t1 --encoder_model_type DLab -b 32