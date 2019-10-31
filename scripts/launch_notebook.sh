#!/bin/bash

module load use.own
module load python/3.7.0
module load pytorch/1.0.0

mkdir -p /tmp/$USER/

function jupyter-launch {
    TARGET=$1

    # Connect SSH
    ssh -N -f -R 0.0.0.0:1617:localhost:1618 $TARGET

    # Launch jupyter
    export XDG_RUNTIME_DIR=''
    jupyter-notebook --no-browser --ip localhost --port 1618 &> /tmp/$USER/error.log &

}

jupyter-launch "$1"
