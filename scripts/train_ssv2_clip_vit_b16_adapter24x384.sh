#!/usr/bin/env sh

torchrun --nproc_per_node 4 main.py \
    --model clip_vit_base_patch16_adapter24x384 \
    --save_dir output_dir/ssv2/clip_vit_base_patch16_adapter24x384 \
    --auto_resume --auto_remove \
    --dataset ssv2 \
    --num_frames 8 \
    --sampling_rate 0 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --batch_size 64 \
    --epochs 50 \
    --warmup_epochs 2 \
    --eval_freq 5
