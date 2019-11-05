import torch.nn.functional as F
from dpn.constants import EPS
from operator import and_
from functools import reduce
from torch.distributions import Dirichlet as Dir
import torch


def entropy_from_logits(logits):
    dimB, dimH = 0, 1
    probs = F.softmax(logits, dim=dimH)
    logprobs = F.log_softmax(logits, dim=dimH)
    plogp = probs*logprobs
    entropy = -1*plogp.sum(dim=dimH)
    return entropy

def max_prob(logits):
    dimB, dimH = 0, 1
    probs = F.softmax(logits, dim=dimH)
    values, indices = torch.max(probs, dim=dimH)
    return values

class Dirichlet:
    def __init__(self, logits=None, probs=None, alphas=None):
        self._logits = logits
        self._probs = probs
        self._alphas = alphas
        self._a0 = None

        # Sanity checking - at least one is defined.
        # flags = [x is None for x in [logits, probs, alphas]]
        # assert(reduce(and_, flags))

        self.dimB, self.dimH = 0, 1

    @property
    def logits(self):
        assert(self._logits is not None)
        return self._logits

    @property
    def alphas(self):
        if self._alphas is None:
            self._alphas = self.logits.exp() + EPS
        return self._alphas

    @property
    def probs(self):
        if self._probs is None:
            self._probs = F.softmax(logits, dim=self.dimH)
        return self._probs

    @property
    def a0(self):
        if self._a0 is None:
            self._a0 =  torch.sum(
                self.alphas, dim=self.dimH, keepdim=True
            )
        return self._a0

    def differential_entropy(self): 
        distribution = Dir(self.alphas)
        return distribution.entropy()

    def expected_entropy(self): 
        alphas = self.alphas
        a0 = self.a0
        expected_entropy = -1 * torch.sum(
                torch.exp(alphas.log() - a0.log())
                * (torch.digamma(alphas + 1.0) - digamma(a0 + 1.0))
            , dim = self.dimH
        )

        return expected_entropy

    def entropy_expected(self): 
        return entropy_from_logits(self.logits)

    def max_prob(self): 
        return max_prob(self.logits)

    def variation_ratio(self): pass

    def mutual_information(self): 
        # Expected entropy seems to be same as mutual information.
        a0 = self.a0
        alphas = self.alphas

        return -1*torch.sum(
            alphas/a0
            * torch.log(alphas/a0)
            - torch.digamma(alphas + 1.0)
            + torch.digamma(a0   + 1.0)
            ,dim = self.dimH
        )


