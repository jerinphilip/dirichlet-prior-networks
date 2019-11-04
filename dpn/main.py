import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dpn.models import build_model
from dpn.criterions import build_criterion
from dpn.args import add_args
from dpn.constants import DatasetType
from collections import namedtuple

def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, samples in enumerate(train_loader):
        optimizer.zero_grad()

        def f(dtype):
            data, labels = samples[dtype]
            data, labels = data.to(device), labels.to(device)
            net_output = model(data)
            in_domain = (dtype == DatasetType.InD)
            _loss = criterion[dtype](net_output, labels, in_domain=in_domain)
            return _loss

        loss = (
                # f(DatasetType.InD) +
                f(DatasetType.OoD)
        )
        # OoD samples
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and args.log:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * train_loader.batch_size, train_loader.num_samples,
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, criterion, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            net_output = model(data)
            logits = net_output['logits']
            test_loss += criterion(net_output, labels)
            pred = logits.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    if args.log:
        print('Epoch {} | Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            epoch,
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def build_optimizer(args, model):
    return optim.Adam(model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)

def build_loader(args):
    from dpn.data import dataset
    return dataset[args.dataset](args)

def main(args):
    loader = build_loader(args)
    device = torch.device(args.device)
    model = build_model(args.model)
    model = model.to(device)
    criterion = build_criterion(args)
    optimizer = build_optimizer(args, model)
    for epoch in range(1, args.epochs + 1):
        train(args, model, criterion, device, loader.train, optimizer, epoch)
        test(args, model, criterion, device, loader.test, epoch)

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    _ = main(args)
