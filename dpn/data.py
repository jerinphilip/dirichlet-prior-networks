
import torch
import math
from torch.utils.data import Dataset
from torchvision import datasets, transforms

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
        return [self.distributions[idx].sample(), torch.Tensor([idx]).long()]

    def construct_distributions(self, radius, sigma):
        def mean(angle):
            return [radius*math.sin(angle), radius*math.cos(angle)]

        angles = [2*math.pi/3, 4*math.pi/3, 6*math.pi/3]
        means = [torch.Tensor(mean(angle)) for angle in angles]

        Normal = torch.distributions.Normal

        distributions = [ Normal(mean, sigma) for mean in means]
        return distributions


transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST
if __name__ == '__main__':
    dataset = SyntheticDataset(3, 1, 100)
    samples = len(dataset)
    for i in range(samples):
        print(dataset[i])

