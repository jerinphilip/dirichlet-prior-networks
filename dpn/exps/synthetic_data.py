import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

from argparse import ArgumentParser

from dpn.main import main
from dpn.args import add_args
from dpn.plotting import plot_entropy, plot_synthetic
from dpn.data import SyntheticDataset
from dpn.functional import entropy_from_logits
from dpn.utils import plt, flush_plot


def inference(model, data):
    data = data.to(torch.device("cuda"))
    net_output = model(data)
    logits = net_output['logits']
    entropy = entropy_from_logits(logits)
    return entropy

def train_routine(radius, sigma):
    parser = ArgumentParser()
    add_args(parser)
    work_dir = '/scratch' # irrelevant for SyntheticDataset
    argv = """
        --dataset synthetic --radius {} --sigma {} \
        --alpha 1e3 \
        --epochs 20 --batch_size 256 --device cuda \
        --momentum 0.9  --lr 1e-3 --weight_decay 0.2 \
        --work_dir {} \
        --model mlp \
    """.format(radius, sigma, work_dir).split()

    args, rest = parser.parse_known_args(argv)
    export = main(args)
    return export

def exp(args):
    def filename_fn(r, s):
        rs = 'radius-{}-sigma-{}.png'.format(r, s)
        fpath = os.path.join(args.output_dir, rs)
        return fpath

    data = SyntheticDataset.grid_data(args.num_points, length=3*args.radius)
    for scale in [4, 3, 2, 1]:
        sigma = scale*args.sigma
        export = train_routine(args.radius, sigma)
        with torch.no_grad():
            entropy = inference(export["model"], data)
            np_x = data.cpu().numpy()
            score = (entropy).exp().cpu().numpy()
            alphas = 1 - 1/score
            normalize = lambda x: (x - np.min(x))/np.ptp(x)
            norm_alphas = normalize(alphas)
            plot_entropy(np_x, norm_alphas)
            plot_synthetic(args.radius, sigma)
            fname = filename_fn(args.radius, sigma)
            flush_plot(plt, fname)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--radius', type=float, required=True)
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--num_points', type=int, required=True)
    args = parser.parse_args()
    exp(args)


