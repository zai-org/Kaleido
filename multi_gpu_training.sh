#! /bin/bash

echo "RUN on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-All}"

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
GPU_COUNT=1

BASE_CONFIG1="configs/video_model/dit_crossattn_14B_wanvae.yaml"
BASE_CONFIG2="configs/training/video_wanx_14B_concat.yaml"
SEED=$RANDOM

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Detected $GPU_COUNT GPUs -> Using torchrun distributed launch"
    run_cmd="torchrun --standalone --nproc_per_node=${GPU_COUNT} train_video_concat.py \
        --base ${BASE_CONFIG1} ${BASE_CONFIG2} \
        --seed ${SEED}"
else
    echo "Detected SINGLE GPU -> Using python single process"
    export LOCAL_RANK=0
    run_cmd="python train_video_concat.py \
        --base ${BASE_CONFIG1} ${BASE_CONFIG2} \
        --seed ${SEED}"
fi

echo "[Command] ${run_cmd}"
eval ${run_cmd}

echo "DONE on $(hostname)"