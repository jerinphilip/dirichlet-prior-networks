import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

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

def exp(args):
    def filename_fn(r, s):
        rs = '{}-radius-{}-sigma-{}.png'.format(args.prefix, r, s)
        fpath = os.path.join(args.output_dir, rs)
        return fpath

    data = SyntheticDataset.grid_data(args.num_points, length=3*args.radius)
    for scale in [4, 3, 2, 1]:
        sigma = scale*args.sigma

        scale_args = deepcopy(args)
        scale_args.sigma = sigma

        export = main(args)
        with torch.no_grad():
            entropy = inference(export["model"], data)
            np_x = data.cpu().numpy()
            score = (entropy).exp().cpu().numpy()
            alphas = 1 - 1/score
            normalize = lambda x: (x - np.min(x))/np.ptp(x)
            norm_alphas = 0.5*normalize(alphas)
            plot_entropy(np_x, norm_alphas)
            plot_synthetic(args.radius, sigma)
            fname = filename_fn(args.radius, sigma)
            flush_plot(plt, fname)


if __name__ == '__main__':
    parser = ArgumentParser()
    add_args(parser)

    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--num_points', type=int, required=True)

    args = parser.parse_args()
    exp(args)


