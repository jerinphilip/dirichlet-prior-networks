#!/bin/bash

module load python/3.7.0
module load pytorch/1.0.0

python3 -m dpn.main \
    --alpha 0.3 \
    --epochs 20 \
    --device cuda \
    --batch_size 256 \
    --momentum 0.9  \
    --lr 1e-3       \
    --work_dir /ssd_scratch/cvit/$USER/ \
    --weight_decay 0.0
