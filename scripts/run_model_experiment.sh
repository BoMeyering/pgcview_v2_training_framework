#!bin/bash

torchrun --standalone --nproc-per-node=2 train_supervised.py --config configs/train_config_unet_resnet18_celoss.yaml --backend nccl

exit 0