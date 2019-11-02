import os
import torch
import hashlib
from tqdm import tqdm, tqdm_notebook

def in_ipynb():
    try:
        cls_name = get_ipython().__class__.__name__ 
        expected = 'ZMQInteractiveShell'
        flag = (cls_name == expected)
        return flag
    except NameError:
        return False

def flush_plot(plt, fname):
    plt.legend()
    plt.savefig(fname)
    if in_ipynb(): 
        plt.show()
    plt.clf()

if in_ipynb():
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt




def hash_args(args):
    args_as_dict = vars(args) 
    params = list(sorted(args_as_dict.items(), key=lambda x: x[0]))
    _hash = hashlib.sha224(params.__repr__().encode()).hexdigest()
    return _hash[:6]


class Saver:
    def __init__(self, directory):
        self.directory = directory
        if not os.path.exists(self.directory):
            print("Creating", self.directory)
            os.makedirs(self.directory)

    def path(self, _hash):
        _path = os.path.join(self.directory, _hash)
        return _path

    def load(self, _hash):
        path = self.path(_hash)
        if os.path.exists(path):
            payload = torch.load(path)
            return payload
        return None

    def save(self, _hash, payload):
        torch.save(payload, self.path(_hash))
        return True

tqdm = tqdm_notebook if in_ipynb() else tqdm
