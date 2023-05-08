#!/usr/bin/env sh

torchrun --nproc_per_node 3 main.py \
    --model clip_vit_base_patch16_adapter12x384 \
    --save_dir output_dir/k400/clip_vit_base_patch16_adapter12x384 \
    --auto_resume --auto_remove \
    --dataset k400 \
    --num_frames 8 \
    --sampling_rate 16 \
    --resize_type random_short_side_scale_jitter \
    --scale_range 1.0 1.15 \
    --num_spatial_views 1 \
    --num_temporal_views 3 \
    --mirror \
    --batch_size 64 \
    --epochs 12 \
    --warmup_epochs 2 \
    --eval_freq 5
