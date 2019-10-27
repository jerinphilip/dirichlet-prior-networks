from collections import namedtuple, defaultdict
from matplotlib import pyplot as plt
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

    loader = dataset['synthetic'](args)
    classwise = defaultdict(list)
    for sample in loader.train:
        points, classes = sample
        points = points.numpy()
        classes = classes.numpy()
        for point, cls in zip(points, classes):
            classwise[cls].append(point.tolist())
    for cls in classwise:
        points = np.array(classwise[cls])
        xs, ys = points[:, 0], points[:, 1]
        idd = {
            "radius": args.radius,
            "sigma": args.sigma
        }
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.title(idd.__repr__())
        plt.scatter(xs, ys, alpha=0.05)
    plt.show()
