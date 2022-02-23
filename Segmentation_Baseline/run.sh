#! /bin/bash
python3 main.py -m t1ce --inference_model SegNet_t1ce_ep100_b256
python3 main.py -m t2 --inference_model SegNet_t2_ep100_b256
python3 main.py -m t1 --inference_model SegNet_t1_ep100_b256