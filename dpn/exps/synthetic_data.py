import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from argparse import ArgumentParser
from ast import literal_eval

from dpn.main import main
from dpn.args import add_args
from dpn.plotting import plot_entropy, plot_synthetic
from dpn.data import SyntheticDataset
from dpn.functional import entropy_from_logits
from dpn.utils import plt, flush_plot, hash_args, Saver, tqdm
from dpn.functional import Dirichlet


def inference(model, data):
    data = data.to(torch.device("cuda"))
    net_output = model(data)
    logits = net_output['logits']

    dirichlet = Dirichlet(logits=logits)
    return dirichlet.differential_entropy()

    entropy = entropy_from_logits(logits)
    return entropy

def exp(args):

    def filename_fn(args):
        losses = literal_eval(args.loss)
        loss_info = [
            '{}-{}'.format(key, value) \
            for key, value in losses.items()
        ]
        lfs = '-'.join(loss_info)
        rs = 'radius-{}-sigma-{}-{}.png'.format(args.radius, args.sigma, lfs)
        return rs

    def fpath(fname):
        _fpath = os.path.join(args.output_dir, fname)
        return _fpath

    length = 5*args.radius
    data = SyntheticDataset.grid_data(args.num_points, length=length)

    plt.xlim(-1*length, length)
    plt.ylim(-1*length, length)

    for scale in tqdm([4, 3, 2, 1]):
        sigma = scale*args.sigma

        scale_args = deepcopy(args)
        scale_args.sigma = sigma
        fname = filename_fn(scale_args)

        checkpoint_dir = os.path.join(args.work_dir, 'checkpoints')
        saver = Saver(checkpoint_dir)
        payload = saver.load(hash_args(scale_args))

        def run_and_save(scale_args):
            export = main(scale_args)
            payload = export['model']
            saver.save(hash_args(scale_args), payload)
            return payload

        export = payload or run_and_save(scale_args)

        with torch.no_grad():
            entropy = inference(export, data)
            np_x = data.cpu().numpy()
            score = (entropy).exp().cpu().numpy()
            # alphas = 1 - 1/score
            alphas = score
            normalize = lambda x: (x - np.min(x))/np.ptp(x)
            norm_alphas = normalize(alphas)
            plot_entropy(np_x, norm_alphas)
            plot_synthetic(scale_args)
            plt.title(fname)
            flush_plot(plt, fpath(fname))


if __name__ == '__main__':
    parser = ArgumentParser()
    add_args(parser)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--num_points', type=int, required=True)
    args = parser.parse_args()
    exp(args)


