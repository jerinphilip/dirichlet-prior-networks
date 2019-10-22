
from torch import nn
import torch
import torch.nn.functional as F


class ConvModel(nn.Module):
    """
    Trying to reproduce the architecture in: 
        https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network_conv.py
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument('--num_filters', type=int, required=True)
        parser.add_argument('--num_conv_layers', type=int, required=True)
        parser.add_argument('--kernel_size', type=int, required=True)
        parser.add_argument('--pool_kernel_size', type=int, required=True)
        parser.add_argument('--output_size', type=int, required=True)
        parser.add_argument('--dropout', type=int, required=True)

    def __init__(
            self, num_filters, num_conv_layers, kernel_size,
            pool_kernel_size, output_size,
            dropout=0.1
    ):
        super().__init__()
        self.num_filters = num_filters
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.pool_kernel_size = pool_kernel_size
        self.dropout = dropout

        def build_conv_bn(in_channels, out_channels, kernel_size):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size)
            bn = nn.BatchNorm2d(out_channels)
            return nn.Sequential(conv, bn)

        self.conv_layers = [
            build_conv_bn( 
                num_filters if i > 0 else 512, num_filters
                kernel_size=self.kernel_size
            ) for i in range(self.num_conv_layers)
        ]

        self.output_size = output_size
        self.fc_out = nn.Linear(conv_output_size, output_size)

    def forward(self, x):
        B, H, W, C = x.size()

        # Work out the Conv math
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
            x = F.maxpool_2d(self.pool_kernel_size)
            x = F.dropout(p=self.dropout, training=self.training)

        x = x.view(B, -1)
        x = self.fc_out(x)

        # x is logits.
        # mean is softmax of logits.
        # alphas are exp of logits.
        # precision is sum of alphas.

        # There's a gain predicted as well.

        return x

class SimpleConvModel(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

