#! /bin/bash
# python3 main.py --encoder_model DLab_t1_ep65_b32 -m t1 --encoder_mode finetuned -b 32
# python3 main.py --encoder_model DLab_t1_ep65_b32 -m t1 --encoder_mode fixed -b 32
# python3 main.py --encoder_model DLab_t1_ep75_b64.unsupervised -m t1 --encoder_mode finetuned -b 32
# python3 main.py -m t1 --encoder_model_type DLab -b 32


# python3 main.py --encoder_model DLab_t2_ep95_b64 -m t2 --encoder_mode finetuned -b 32
# python3 main.py --encoder_model DLab_t2_ep95_b64 -m t2 --encoder_mode fixed -b 32

# python3 main.py -m t1ce --encoder_model_type DLab -b 32
python3 main.py --encoder_model DLab_t1ce_ep50_b64 -m t1ce --encoder_mode finetuned -b 32