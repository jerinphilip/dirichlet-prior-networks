#!/bin/bash

module load python/3.7.0
module load pytorch/1.0.0

FILES=(
    cifar-10.tar
    mnist.tar
    svhn.tar
)

TORCHVISION_PATH="ada:/share1/dataset/torchvision"
WORK_DIR="/ssd_scratch/cvit/$USER/dpn"
mkdir -p $WORK_DIR

set -x

function copy {
    for FILE in ${FILES[@]}; do
        rsync $TORCHVISION_PATH/$FILE $WORK_DIR/
        cd $WORK_DIR && tar -xvf $FILE && cd -;
    done;
}

# copy

python3 -m dpn.main \
    --dataset mnist \
    --alpha 1e3 \
    --epochs 1000 \
    --device cuda \
    --batch_size 256 \
    --momentum 0.9  \
    --lr 1e-3       \
    --work_dir $WORK_DIR \
    --weight_decay 0.0 \
    --model vgg6
