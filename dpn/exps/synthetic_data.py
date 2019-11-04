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
from dpn.plotting import plot_entropy, plot_synthetic, plot_pcolormesh
from dpn.data import SyntheticDataset
from dpn.functional import entropy_from_logits
from dpn.utils import plt, flush_plot, hash_args, Saver, tqdm
from dpn.functional import Dirichlet


def inference(model, data):
    data = data.to(torch.device("cuda"))
    net_output = model(data)
    logits = net_output['logits']
    dirichlet = Dirichlet(logits=logits)
    differential_entropy = dirichlet.differential_entropy()
    mutual_information = dirichlet.mutual_information()
    entropy = entropy_from_logits(logits)
    export =  {
        "entropy": entropy, 
        "mutual_information": mutual_information, 
        "differential_entropy": differential_entropy
    }
    return export

def exp(args):

    def filename_fn(args):
        # losses = literal_eval(args.ind_loss)
        # loss_info = [
        #     '{}-{}'.format(key, value) \
        #     for key, value in losses.items()
        # ]
        # lfs = '-'.join(loss_info)
        rs = 'N({}, {})'.format(args.radius, args.sigma)
        return rs

    def fpath(fname):
        _fpath = os.path.join(args.output_dir, fname)
        return _fpath

    length = 5*args.radius
    linspace, data = SyntheticDataset.grid_data(args.num_points, length=length)

    plt.xlim(-1*length, length)
    plt.ylim(-1*length, length)

    for scale in tqdm([1, 2, 3, 4]):
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
            scores = inference(export, data)
            np_x = data.cpu().numpy()
            for key in scores:
                score = scores[key].cpu().numpy()
                plot_pcolormesh(np_x, linspace, score)
                score_fname = '{}_{}'.format(fname, key)
                plt.title(score_fname)
                flush_plot(plt, fpath(score_fname) + '.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    add_args(parser)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--num_points', type=int, required=True)
    args = parser.parse_args()
    exp(args)


