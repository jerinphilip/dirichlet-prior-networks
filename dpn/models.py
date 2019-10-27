
from torch import nn
import torch
import torch.nn.functional as F


class ConvModel(nn.Module):
    """
    Trying to reproduce the architecture in: 
        https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network_conv.py
    """
    def __init__(
            self, num_filters, kernel_size,
            pool_kernel_size, fc_layer_sizes, output_classes,
            dropout=0.1, gain=False
    ):
        super().__init__()
        self.num_filters = num_filters
        self.fc_layer_sizes = fc_layer_sizes
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.pool_kernel_size = pool_kernel_size
        self.dropout = dropout
        self.output_classes = output_classes
        self.gain = gain

        self.conv_layers = self.build_conv_layers(num_filters, kernel_size)
        self.fc_layers = self.build_fc_layers(fc_layer_sizes)
        self.fc_out = nn.Linear(self.fc_layer_sizes[-1], output_classes)

        if self.gain:
            self.fc_gain = nn.Linear(self.fc_layer_sizes[-1], 1)

    def build_conv_layers(self, num_filters, kernel_size):
        def build_conv_bn(in_channels, out_channels, kernel_size):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size)
            bn = nn.BatchNorm2d(out_channels)
            return nn.Sequential(conv, bn)

        conv_layers = [
            build_conv_bn( 
                num_filters[i], num_filters[i+1],
                kernel_size=kernel_size
            ) for i in range(len(num_filters) - 1)
        ]

        conv_layers = nn.ModuleList(conv_layers)
        return conv_layers

    def build_fc_layers(self, fc_layer_sizes):
        def build_fc_bn(input_size, output_size):
            fc = nn.Linear(input_size, output_size)
            bn = nn.BatchNorm1d(output_size)
            return nn.Sequential(fc, bn)

        fc_layers = [
                build_fc_bn(fc_layer_sizes[i], fc_layer_sizes[i+1])
                for i in range(len(fc_layer_sizes)-1)
        ]


        fc_layers = nn.ModuleList(fc_layers)
        return fc_layers

    def forward(self, x):
        B, H, W, C = x.size()

        # Work out the Conv math
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=self.pool_kernel_size, padding=1)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(B, -1)

        for fc in self.fc_layers:
            x = fc(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.fc_out(x)
        gain = 0 if not self.gain else F.relu(self.fc_gain(x))

        return {
            'logits': logits,
            'gain': gain
        }


def build_model(tag):
    constructors = {
        'vgg6': (
            lambda: ConvModel(
                num_filters=[1, 16, 32, 32, 32], 
                # num_conv_layers=4, 
                kernel_size=(3, 3),
                pool_kernel_size=(2, 2), 
                output_classes=10,
                dropout=0.5,
                fc_layer_sizes=[32, 100],
                gain=False
            )
        ),
    }

    return constructors[tag]()


class SimpleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
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

class VanillaMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.network = nn.Sequential(
           nn.Linear(input_size, hidden_size),
           nn.Sigmoid(),
           nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return { 'logits': self.network(x), 'gain': 0.0}

