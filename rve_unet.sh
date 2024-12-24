# config_file_path1="configs/rve_unet.yaml"
# name1="rve_unet_64x64_nograd"
# python train.py --config $config_file_path1 --name $name1 --seed 101 --logdir logs \
# --image_logger_disable

config_file_path2="configs/rve_unet_grad.yaml"
name2="rve_unet_64x64_grad"
python train.py --config $config_file_path2 --name $name2 --seed 101 --logdir logs \
--image_logger_disable

config_file_path3="configs/rve_unet_grad_share.yaml"
name3="rve_unet_64x64_grad_share"
python train.py --config $config_file_path3 --name $name3 --seed 101 --logdir logs \
--image_logger_disable
