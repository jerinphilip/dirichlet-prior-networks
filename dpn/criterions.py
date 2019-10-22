

from torch import nn
import torch 
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

class ExpectedKL(nn.Module):
    def __init__(self, num_labels, eps=1e-8):
        super().__init__()
        self.eps = 1e-8
        self.num_labels = num_labels

    def forward(self, logits, labels):
        # Translating below segment to PyTorch. This one is present in  the paper.
        # https://github.com/KaosEngineer/PriorNetworks-OLD/blob/
        # 79cb8300238271566a5bbb69f0744f1d80924a1a/
        # prior_networks/dirichlet/dirichlet_prior_network.py
        # #L328-L334

        def one_hot(labels):
            # Credits: ptrblck
            # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/31

            x = torch.zeros(len(labels), self.num_labels)
            x.scatter_(1, labels.unsqueeze(1), 1.)
            return x


        targets = one_hot(labels)

        B, H = logits.size()
        dimB, dimH = 0, 1
        alphas = logits.exp() + self.eps
        concentration =  alphas.sum(axis = dimH)
        loss = (
                concentration.lgamma() 
                - alphas.lgamma.sum(dim = dimH) 
                + torch.sum(
                    (alphas - 1.0) * targets.float().log(),
                    dim = dimH
                )
        )

        return loss



