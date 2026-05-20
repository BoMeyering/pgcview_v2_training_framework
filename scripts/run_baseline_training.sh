#!bin/bash

configs=(
    "configs/baseline/segformer_enb1_celoss.yaml"
    "configs/baseline/segformer_enb1_weighted_celoss.yaml"
    "configs/baseline/segformer_enb1_weighted_focalloss.yaml"
    "configs/baseline/segformer_enb1_cbceloss.yaml"
    "configs/baseline/segformer_enb1_cbfocalloss.yaml"
    "configs/baseline/segformer_enb1_acbceloss.yaml"
    "configs/baseline/segformer_enb1_recallceloss.yaml"
    "configs/baseline/segformer_enb1_tvmfdiceloss_k=16.yaml"
    "configs/baseline/segformer_enb1_tvmfdiceloss_lam=32.yaml"
    "configs/baseline/segformer_enb2_tvmfdiceloss_k=16.yaml"
    "configs/baseline/segformer_enb3_tvmfdiceloss_k=16.yaml"
    "configs/baseline/segformer_enb4_tvmfdiceloss_k=16.yaml"
    "configs/baseline/segformer_enb5_tvmfdiceloss_k=16.yaml"
    "configs/baseline/segformer_mitb1_tvmfdiceloss_k=16.yaml"
    "configs/baseline/segformer_mitb2_tvmfdiceloss_k=16.yaml"
)

for cfg in "${configs[@]}"; do
    torchrun --standalone --nproc-per-node=2 train_supervised.py --config "$cfg" --backend nccl
done

exit 0