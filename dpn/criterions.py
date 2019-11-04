from torch import nn
import torch 
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from collections import namedtuple
from dpn.constants import EPS, DatasetType
from ast import literal_eval


def one_hot(labels, num_labels):
    # Credits: ptrblck
    # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/31

    x = labels.new_zeros((len(labels), num_labels))
    x.scatter_(1, labels.unsqueeze(1), 1.)
    return x

def label_smooth(labels_one_hot, smoothing):
    if smoothing < EPS:
        return labels_one_hot

    batch_size, num_classes = labels_one_hot.size()
    smoothed = (
            (1 - num_classes*smoothing)*labels_one_hot 
            + smoothing * torch.ones_like(labels_one_hot)
    )
    return smoothed

def dirichlet_params_from_logits(logits):
    dimB, dimH = 0, 1
    alphas = logits.exp() + EPS
    concentration = alphas.sum(dim = dimH, keepdim=True)
    return alphas, concentration


def lgamma(tensor):
    # Some confusion with lgamma's missing documentation.
    return torch.lgamma(tensor)


class Cost(nn.Module):
    @classmethod
    def build(cls, args):
        return cls()

class NLLCost(Cost):
    def __init__(self, smoothing=1e-2, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.reduce = reduce

    def forward(self, net_output, labels, **kwargs):
        # Translating below segment to PyTorch. This one is present in  the paper.
        # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/79cb8300238271566a5bbb69f0744f1d80924a1a/prior_networks/dirichlet/dirichlet_prior_network.py#L328-L334

        logits = net_output['logits']
        batch_size, num_classes = logits.size()
        labels_one_hot = one_hot(labels, num_classes)
        targets = label_smooth(labels_one_hot, self.smoothing)
        dimB, dimH = 0, 1

        alphas, concentration = dirichlet_params_from_logits(logits)

        loss = (
                lgamma(concentration)
                - lgamma(alphas).sum(dim = dimH) 
                + torch.sum(
                    (alphas - 1.0) * targets.float().log(),
                    dim = dimH
                )
        )

        return loss.mean()

class CrossEntropy(Cost):
    def __init__(self):
        super().__init__()

    def forward(self, net_output, labels, **kwargs):
        logits = net_output['logits']
        return F.cross_entropy(logits, labels)

class DirichletKLDiv(Cost):
    def __init__(self, alpha, reduce=True, smoothing=1e-2):
        super().__init__()
        self.alpha = alpha
        self.reduce = reduce
        self.smoothing = smoothing

    def forward(self, net_output, labels, in_domain=True, **kwargs):
        # Translation of
        # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network.py#L281-L294

        logits, gain = net_output['logits'], net_output['gain']

        batch_size, num_classes = logits.size()
        dimB, dimH = 0, 1

        # mean and precision from the network
        mean = F.softmax(logits, dim=dimH)
        precision = torch.sum((logits + gain).exp(), dim=dimH, keepdim=True)

        # the expected mean and precision, from the ground truth
        labels_one_hot = one_hot(labels, num_classes).float()

        def in_domain_targets():
            target_mean = label_smooth(labels_one_hot, self.smoothing)
            target_precision = self.alpha * precision.new_ones((batch_size, 1))
            return target_mean, target_precision

        def out_of_domain_targets():
            # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network.py#L619-L623
            target_precision = num_classes * precision.new_ones((batch_size, 1)).float()
            target_mean = torch.ones_like(mean).float()/num_classes
            return target_mean, target_precision

        target_f = in_domain_targets if in_domain else out_of_domain_targets
        target_mean, target_precision = target_f()

        loss = self._compute_loss(mean, precision, target_mean, target_precision)
        return loss

    def _compute_loss(self, mean, precision, target_mean, target_precision):
        eps = EPS
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

    @classmethod
    def build(cls, args):
        return cls(args.alpha)

class MutualInformation(Cost):
    def __init__(self):
        super().__init__()

    def forward(self, net_output, labels, **kwargs):
        logits = net_output['logits']
        dimB, dimH = 0, 1
        alphas, concentration = dirichlet_params_from_logits(logits)

        mutual_information = torch.sum(
                alphas/concentration * 
                (
                    torch.log(alphas) 
                    - torch.log(concentration)
                    - torch.digamma(alphas + 1.0)
                    + torch.digamma(concentration + 1.0)
                )
            ,dim=dimH
        )

        loss = mutual_information.mean()
        return loss

class DifferentialEntropy(Cost):
    def __init__(self):
        super().__init__()

    def forward(self, net_output, labels, **kwargs):
        logits = net_output['logits']

        dimB, dimH = 0, 1
        batch_size, num_classes = logits.size()

        alphas, concentration = dirichlet_params_from_logits(logits)

        differential_entropy = (
            torch.sum(lgamma(alphas) , dim = dimH)
             - lgamma(concentration)
             + (concentration - num_classes) * torch.digamma(concentration)
             - torch.sum((alphas - 1.0) * torch.digamma(alphas), dim=dimH)
        )

        loss = differential_entropy.mean()
        return loss

class MultiTaskLoss(nn.Module):
    def __init__(self, weighted_losses):
        super().__init__()
        self.losses = weighted_losses

    def forward(self, net_output, labels, **kwargs):
        accumulator = 0
        for loss in self.losses:
            if loss.weight:
                accumulator += loss.weight * loss.f(net_output, labels, **kwargs)
        return accumulator


def build_criterion(args):
    # Switch to control criterion
    # Criterion is a multi-task-objective.
    # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/master/prior_networks/dirichlet/dirichlet_prior_network.py#L629-L640

    loss_functions = {
        "dirichlet_kldiv": DirichletKLDiv,
        "cross_entropy": CrossEntropy,
        "differential_entropy": DifferentialEntropy,
        "nll": NLLCost,
        "mutual_information": MutualInformation
    }

    def validate_keys(weights):
        wkeys = set(list(weights.keys()))
        lkeys = set(list(loss_functions.keys()))
        assert((wkeys <= lkeys), "Check loss supplied")

    def build_weighted_loss(weights):
        WeightedLoss = namedtuple('WeightedLoss', 'weight f')
        weighted_losses = [
            WeightedLoss(weight=weights[fname], f=loss_functions[fname].build(args))
            for fname in weights.keys()
        ]
        criterion = MultiTaskLoss(weighted_losses)
        return criterion

    ind_weights = literal_eval(args.ind_loss)
    ood_weights = literal_eval(args.ood_loss)

    validate_keys(ind_weights)
    validate_keys(ood_weights)

    criterion = {
        DatasetType.InD: build_weighted_loss(ind_weights),
        DatasetType.OoD: build_weighted_loss(ood_weights)
    }

    return criterion
