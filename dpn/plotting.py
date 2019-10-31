from collections import namedtuple, defaultdict
from dpn.utils import plt
import numpy as np
from dpn.data import dataset


def plot_synthetic(radius, sigma):
    Args = namedtuple('Args', 'radius sigma num_train_samples num_test_samples batch_size shuffle')
    args = Args(
        radius=radius, sigma=sigma, 
        num_train_samples=int(1e4), num_test_samples=int(1e3),
        batch_size=400,
        shuffle=False
    )

    params = {
        "radius": args.radius,
        "sigma": args.sigma
    }

    loader = dataset['synthetic'](args)
    classwise = defaultdict(list)
    for sample in loader.train:
        points, classes = sample
        points = points.numpy()
        classes = classes.numpy()
        for point, cls in zip(points, classes):
            classwise[cls].append(point.tolist())
    handle = plot_synthetic_classwise(params.__repr__(), classwise)
    return handle

def plot_synthetic_classwise(title, classwise):
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.title(title)
    for cls in classwise:
        points = np.array(classwise[cls])
        xs, ys = points[:, 0], points[:, 1]
        plt.scatter(xs, ys, alpha=0.05)
    return plt

def plot_entropy(np_x, scores):
    scale = 100
    num_points, _ = np_x.shape
    rgba_colors = np.zeros((num_points, 4))
    rgba_colors[:, :3] = np.array([0.4, 0.4, 1.0])
    # rgba_colors[:, 3] = (1 - 1/score)
    rgba_colors[:, 3] = scores
    xs, ys = np_x[:, 0], np_x[:, 1]
    plt.scatter(xs, ys, color=rgba_colors)

