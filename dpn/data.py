import os
import torch
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from collections import namedtuple
from dpn.constants import DatasetType

Split = namedtuple('Split', 'train dev test')
transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
])

class SyntheticDataset(Dataset):
    def __init__(self, radius, sigma, num_samples):
        self.sigma = sigma
        self.radius = radius
        self.num_samples = num_samples
        self.distributions = self.construct_distributions(radius, sigma)

    def __len__(self):
        return 3*self.num_samples

    def __getitem__(self, idx):
        idx = idx % 3
        return [self.distributions[idx].sample(), idx]

    def construct_distributions(self, radius, sigma):
        def mean(angle):
            return [radius*math.sin(angle), radius*math.cos(angle)]

        angles = [2*math.pi/3, 4*math.pi/3, 6*math.pi/3]
        means = [torch.Tensor(mean(angle)) for angle in angles]

        Normal = torch.distributions.Normal
        MVNormal = torch.distributions.MultivariateNormal

        # distributions = [ Normal(mean, sigma) for mean in means]
        distributions = [ MVNormal(mean, sigma*torch.eye(2)) for mean in means]
        return distributions

    @staticmethod
    def grid_data(num_points, length=None):
        length = length or 2*self.radius
        ps = torch.linspace(-1*length, length, num_points)
        x = torch.zeros((num_points, num_points, 2))
        for i in range(num_points):
            for j in range(num_points):
                x[i, j, :] = torch.Tensor([ps[i], ps[j]])
        x = x.view(-1, 2)
        return ps, x

class RandomNoise2D(Dataset):
    def __init__(self, synthetic_dataset, num_samples, threshold=1e-3):
        self.synthetic_dataset = synthetic_dataset
        self.num_samples = num_samples
        self.threshold = threshold

        radius = synthetic_dataset.radius
        sigma = synthetic_dataset.sigma
        length = 5 * sigma

        Uniform = torch.distributions.Uniform
        minval, maxval = -1*length, length
        low = torch.Tensor([minval, minval])
        high = torch.Tensor([maxval, maxval])
        self.distribution = Uniform(low, high)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = None
        prob = 1.0

        while prob > self.threshold:
            sample = self.distribution.sample()
            individual_probs = [dist.log_prob(sample).exp().item()
                    for dist in self.synthetic_dataset.distributions]
            prob = torch.Tensor(individual_probs).float().mean().item()

        return (sample, -1)


class DPNLoader:
    def __init__(self, ind_data, ood_data, ind_fraction, batch_size=1, **kwargs):
        self.batch_size = batch_size

        in_batch_size = math.floor(ind_fraction*batch_size)
        out_batch_size = math.ceil((1 - ind_fraction)*batch_size)

        in_batch_size = int(in_batch_size) 
        out_batch_size = int(out_batch_size)

        self.num_samples = min(len(ind_data), len(ood_data))

        self.ind_dataloader = DataLoader(ind_data, batch_size=in_batch_size, **kwargs)
        self.ood_dataloader = DataLoader(ood_data, batch_size=out_batch_size, **kwargs)

    def __len__(self):
        return min(len(self.ind_dataloader), len(self.ood_dataloader))

    def __iter__(self):
        self.ind_iter = iter(self.ind_dataloader)
        self.ood_iter = iter(self.ood_dataloader)
        return self

    def __next__(self):
        return {
            DatasetType.InD: next(self.ind_iter),
            DatasetType.OoD: next(self.ood_iter)
        }

def create_loader(split, args):
    train_loader = torch.utils.data.DataLoader( 
        split.train, batch_size=args.batch_size, 
        shuffle=args.shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        split.test, batch_size=args.batch_size, 
        shuffle=False
    )
    loader = Split(train=train_loader, dev=[], test=test_loader)
    return loader

def create_dpn_loader(Ind, Ood):
    def build_with_args(args):
        loaders = {}
        ind = Ind(args)
        ood = Ood(args)
        for attr in ['train', 'test', 'dev']:
            ind_split = getattr(ind, attr)
            ood_split = getattr(ood, attr)
            if ind_split is not None and ood_split is not None:
                shuffle = args.shuffle if (attr == 'train') else False
                loader = DPNLoader(
                    ind_split, ood_split, ind_fraction=args.ind_fraction,
                    batch_size = args.batch_size, 
                    shuffle = shuffle
                )
            loaders[attr] = loader
        return Split(**loaders)
    return build_with_args

def MNIST(args):
    work_dir = os.path.join(args.work_dir, 'mnist')
    train_dataset = datasets.MNIST(work_dir, train=True, 
            download=False, transform=transform)
    test_dataset = datasets.MNIST(work_dir, train=False, transform=transform),
    return Split(train=train_dataset, dev=None, test=test_dataset)

def Synthetic(args):
    def gen(num_samples):
        synthetic_dataset = SyntheticDataset(radius=args.radius, 
                sigma=args.sigma, num_samples=num_samples)
        return synthetic_dataset

    train_dataset = gen(args.num_train_samples)
    test_dataset = gen(args.num_test_samples)
    return Split(train=train_dataset, dev=None, test=test_dataset)

def RandomNoise2DBuilder(args):
    synthetic_dataset = SyntheticDataset(radius=args.radius, 
                sigma=args.sigma, num_samples=0)

    def gen(num_samples):
        return RandomNoise2D(synthetic_dataset, num_samples=args.num_train_samples)

    train_dataset = gen(args.num_train_samples)
    test_dataset = gen(args.num_test_samples)
    return Split(train=train_dataset, dev=None, test=test_dataset)


def build_loader(dataset_c):
    def build_using_args(args):
        dataset_split = dataset_c(args)
        return create_loader(dataset_split, args)
    return build_using_args

dataset = {
    'mnist': create_dpn_loader(MNIST, MNIST),
    'synthetic': create_dpn_loader(Synthetic, RandomNoise2DBuilder),
}


# MNIST
if __name__ == '__main__':
    dataset = SyntheticDataset(3, 1, 100)
    samples = len(dataset)
    for i in range(samples):
        print(dataset[i])

