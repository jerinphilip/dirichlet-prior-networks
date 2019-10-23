import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dpn.models import build_conv_model
from dpn.criterions import ExpectedKL, NLLCost, CrossEntropyLoss, MultiTaskLoss
from dpn.args import add_args
from collections import namedtuple

def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        net_output = model(data)
        loss = criterion(net_output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            net_output = model(data)
            logits = net_output['logits']
            # test_loss += F.nll_loss(logits, labels, reduction='sum').item() # sum up batch loss
            test_loss += criterion(net_output, labels)
            pred = logits.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.work_dir, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.work_dir, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)



    device = torch.device(args.device)
    model = build_conv_model(args.model)
    model = model.to(device)

    WeightedLoss = namedtuple('WeightedLoss', 'weight f')
    weighted_losses = [
        WeightedLoss(weight=1.0, f=ExpectedKL(alpha = args.alpha)),
        WeightedLoss(weight=0.0, f=CrossEntropyLoss()),
    ]
    criterion = MultiTaskLoss(weighted_losses)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        train(args, model, criterion, device, train_loader, optimizer, epoch)
        test(args, model, criterion, device, test_loader)


