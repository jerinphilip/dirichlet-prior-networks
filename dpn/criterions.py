from torch import nn
import torch 
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from collections import namedtuple

def one_hot(labels, num_labels):
    # Credits: ptrblck
    # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/31

    x = labels.new_zeros((len(labels), num_labels))
    x.scatter_(1, labels.unsqueeze(1), 1.)
    return x

def label_smooth(labels_one_hot, smoothing):
    B, H = labels_one_hot.size()
    smoothed = (
            (1 - H*smoothing)*labels_one_hot 
            + smoothing * torch.ones_like(labels_one_hot)
        )
    return smoothed


def lgamma(tensor):
    # Some confusion with lgamma's missing documentation.
    return torch.lgamma(tensor)

class NLLCost(nn.Module):
    def __init__(self, smoothing=1e-2, eps=1e-8, reduce=True):
        super().__init__()
        self.eps = 1e-8
        self.smoothing = smoothing
        self.reduce = reduce

    def forward(self, net_output, labels):
        # Translating below segment to PyTorch. This one is present in  the paper.
        # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/79cb8300238271566a5bbb69f0744f1d80924a1a/prior_networks/dirichlet/dirichlet_prior_network.py#L328-L334

        logits = net_output['logits']
        B, H = logits.size()
        labels_one_hot = one_hot(labels, H)
        targets = (labels_one_hot if self.smoothing > self.eps else label_smooth(labels_one_hot, self.smoothing))
        B, H = logits.size()
        dimB, dimH = 0, 1
        alphas = logits.exp() + self.eps
        concentration =  alphas.sum(axis = dimH)
        loss = (
                lgamma(concentration)
                - lgamma(alphas).sum(dim = dimH) 
                + torch.sum(
                    (alphas - 1.0) * targets.float().log(),
                    dim = dimH
                )
        )

        # In the end, this looks like a complicated formulation of a
        # probability using dirichlet components.
        # logits can be trained with crossentropy, which should be
        # equivalent, why are we going through this pain?

        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, net_output, labels):
        logits = net_output['logits']
        return F.cross_entropy(logits, labels)

class DirichletKLDiv(nn.Module):
    def __init__(self, alpha, eps=1e-8, reduce=True, smoothing=1e-2):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduce = reduce
        self.smoothing = smoothing

    def forward(self, net_output, labels):
        # TODO(jerin): Make one more pass. There may be a more
        # numerically stable way of doing this.

        # Translation of
        # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network.py#L281-L294

        logits, gain = net_output['logits'], net_output['gain']

        B, H = logits.size()
        dimB, dimH = 0, 1

        mean = F.softmax(logits, dim=dimH)
        precision = torch.sum((logits + gain).exp(), dim=dimH, keepdim=True)
        labels_one_hot = one_hot(labels, H).float()
        target_precision = self.alpha * precision.new_ones((B, 1)).float()

        if abs(self.smoothing) > self.eps:
            target_mean = label_smooth(labels_one_hot, self.smoothing)
        else:
            target_mean = labels_one_hot

        loss = self._compute_loss(mean, precision, target_mean, target_precision)
        return loss

    def _compute_loss(self, mean, precision, target_mean, target_precision):
        eps = self.eps
        B, H = mean.size()
        dimB, dimH = 0, 1
        dlgamma = lgamma(target_precision + eps) - lgamma(precision + eps)
        dsumlgamma = torch.sum(
            (lgamma(mean * precision + eps)
              - lgamma(target_mean * target_precision + eps)
            ), dim = dimH
        )

        dconc = target_precision * target_mean - precision * mean
        dphiconc = (
            torch.digamma(target_mean * target_precision + eps) 
            - torch.digamma(target_precision + eps)
        )

        dprodconc_conc_phi = torch.sum(dconc*dphiconc, dim = dimH)
        loss = (dlgamma + dsumlgamma + dprodconc_conc_phi)
        loss = loss.mean()
        return loss

class MultiTaskLoss(nn.Module):
    def __init__(self, weighted_losses):
        super().__init__()
        self.losses = weighted_losses

    def forward(self, net_output, labels):
        accumulator = 0
        for loss in self.losses:
            if loss.weight:
                accumulator += loss.weight * loss.f(net_output, labels)
        return accumulator


def build_criterion(args):
    # Switch to control criterion
    # Criterion is a multi-task-objective.
    # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network.py#L629-L640
    WeightedLoss = namedtuple('WeightedLoss', 'weight f')
    weighted_losses = [
        WeightedLoss(weight=1.0, f=DirichletKLDiv(alpha = args.alpha)),
        WeightedLoss(weight=1.0, f=CrossEntropyLoss()),
    ]
    criterion = MultiTaskLoss(weighted_losses)
    return criterion
