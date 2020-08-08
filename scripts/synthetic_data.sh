#!/bin/bash

module load python/3.7.0
module load pytorch/1.0.0

WORK_DIR="/ssd_scratch/cvit/jerin/dpn-exps"
mkdir -p $WORK_DIR

function f {
    python3 -m dpn.exps.synthetic_data \
        --alpha 1e3 --epochs 10 --device cuda \
        --batch_size 1024 --momentum 0.9 --lr 1e-3 \
        --weight_decay 0.0 --work_dir $WORK_DIR \
        --model mlp --dataset synthetic --radius 8.0 \
        --sigma 2.5 --shuffle \
        --num_train_samples 100000 --num_test_samples 1000 \
        --log --ind-loss "$1" --ood-loss "$2"\
        --output-dir $WORK_DIR  \
        --num_points 200 --log_interval 10 \
        --rejection-threshold 1e-5
        # --rejection-threshold 1.0672106745950928e-05 # norm.pdf(3.1)**2
}

# f "{'cross_entropy': 1.0}"
f "{'dirichlet_kldiv': 1.0, 'cross_entropy': 1.0}" "{'dirichlet_kldiv': 1.0}"
# f "{'dirichlet_kldiv': 1.0}" "{'dirichlet_kldiv': 1.0}"
# f "{'nll': 1.0, 'mutual_information': 1.0}"
