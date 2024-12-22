#!/bin/bash

# config_file_path1="configs/rve_unconditional_reg.yaml"
# name1="rve_unconditional_reg_64x64"
# python train.py --config $config_file_path1 --name $name1 --seed 101 --logdir logs

config_file_path2="configs/rve_unconditional.yaml"
name2="rve_unconditional_64x64"
python train.py --config $config_file_path2 --name $name2 --seed 101 --logdir logs
