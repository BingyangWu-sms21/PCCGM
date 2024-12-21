#!/bin/bash

config_file_path="configs/rve_unconditional.yaml"
name="rve_unconditional_64x64"
python train.py --config $config_file_path --name $name --seed 101 --logdir logs
