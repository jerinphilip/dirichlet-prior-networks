import os
import torch
import math
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from collections import namedtuple

Loader = namedtuple('Loader', 'train dev test')
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

        distributions = [ Normal(mean, sigma) for mean in means]
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
        return x


def create_loader(train_dataset, dev_dataset, test_dataset, args):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=args.shuffle)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False)
    loader = Loader(train=train_loader, dev=[], test=test_loader)
    return loader

def MNIST(args):
    work_dir = os.path.join(args.work_dir, 'mnist')
    train_dataset = datasets.MNIST(work_dir, train=True, 
            download=False, transform=transform)
    test_dataset = datasets.MNIST(work_dir, train=False, transform=transform),
    return create_loader(train_dataset, None, test_dataset, args)

def Synthetic(args):
    def gen(num_samples):
        dataset = SyntheticDataset(radius=args.radius, 
                sigma=args.sigma, num_samples=num_samples)
        return dataset

    train_dataset = gen(args.num_train_samples)
    test_dataset = gen(args.num_test_samples)
    return create_loader(train_dataset, None, test_dataset, args)

dataset = {
    'mnist': MNIST,
    'synthetic': Synthetic
}


# MNIST
if __name__ == '__main__':
    dataset = SyntheticDataset(3, 1, 100)
    samples = len(dataset)
    for i in range(samples):
        print(dataset[i])

