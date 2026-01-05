#!bin/bash

configs=(
    "configs/loss_experiment/segformer_enb1_celoss.yaml"
    "configs/loss_experiment/segformer_enb1_focalloss.yaml"
    "configs/loss_experiment/segformer_enb1_weighted_celoss.yaml"
    "configs/loss_experiment/segformer_enb1_weighted_focalloss.yaml"
    "configs/loss_experiment/segformer_enb1_cbceloss.yaml"
    "configs/loss_experiment/segformer_enb1_cbfocalloss.yaml"
    "configs/loss_experiment/segformer_enb1_acbceloss.yaml"
    "configs/loss_experiment/segformer_enb1_acbfocalloss.yaml"
    "configs/loss_experiment/segformer_enb1_recallceloss.yaml"
    "configs/loss_experiment/segformer_enb1_recallfocalloss.yaml"
    "configs/loss_experiment/segformer_enb1_diceloss.yaml"
    "configs/loss_experiment/segformer_enb1_focal1.5diceloss.yaml"
    "configs/loss_experiment/segformer_enb1_focal2.0diceloss.yaml"
    "configs/loss_experiment/segformer_enb1_tverskyloss.yaml"
    "configs/loss_experiment/segformer_enb1_tvmfdiceloss_k=16.yaml"
    "configs/loss_experiment/segformer_enb1_tvmfdiceloss_k=32.yaml"
    "configs/loss_experiment/segformer_enb1_tvmfdiceloss_k=64.yaml"
    "configs/loss_experiment/segformer_enb1_tvmfdiceloss_lam=2.yaml"
    "configs/loss_experiment/segformer_enb1_tvmfdiceloss_lam=32.yaml"
    "configs/loss_experiment/segformer_enb1_tvmfdiceloss_lam=128.yaml"
)

for cfg in "${configs[@]}"; do
    torchrun --standalone --nproc-per-node=2 train_supervised.py --config "$cfg" --backend nccl
done

exit 0