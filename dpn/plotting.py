from collections import namedtuple, defaultdict
from dpn.utils import plt
import numpy as np
from dpn.data import dataset


def plot_synthetic(args):
    loader = dataset['synthetic'](args)
    classwise = defaultdict(list)
    for sample in loader.train:
        points, classes = sample
        points = points.numpy()
        classes = classes.numpy()
        for point, cls in zip(points, classes):
            classwise[cls].append(point.tolist())
    handle = plot_synthetic_classwise(classwise)
    return handle

def plot_synthetic_classwise(classwise):
    for i, cls in enumerate(classwise):
        label = 'class-{}'.format(i+1)
        points = np.array(classwise[cls])
        xs, ys = points[:, 0], points[:, 1]
        plt.scatter(xs, ys, alpha=0.1, label=label)
    return plt

def plot_entropy(np_x, scores):
    scale = 100
    num_points, _ = np_x.shape
    rgba_colors = np.zeros((num_points, 4))
    rgba_colors[:, :3] = np.array([0.4, 0.4, 1.0])
    # rgba_colors[:, 3] = (1 - 1/score)
    rgba_colors[:, 3] = scores
    xs, ys = np_x[:, 0], np_x[:, 1]
    plt.scatter(xs, ys, color=rgba_colors, label='entropy')

