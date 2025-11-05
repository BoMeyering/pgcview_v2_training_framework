#!bin/bash

configs=(
    "configs/model_experiment/unet_enb1.yaml"
    "configs/model_experiment/unetpp_enb1.yaml"
    "configs/model_experiment/pan_enb1.yaml"
    "configs/model_experiment/fpn_enb1.yaml"
    "configs/model_experiment/dlv3_enb1.yaml"
    "configs/model_experiment/dlv3p_enb1.yaml"
    "configs/model_experiment/segformer_enb1.yaml"
)

for cfg in "${configs[@]}"; do
    torchrun --standalone --nproc-per-node=2 train_supervised.py --config "$cfg" --backend nccl
done

exit 0