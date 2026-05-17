#!/bin/bash
python inference.py --dataset cifar10 --model vgg8 --hardware 1 --bitcell 1 --sub_array "[128,128]" --mem_type "resistive" --off_state 1e-15 --on_state 1e-14 --adc_precision 7 --data_path ./datasets/

