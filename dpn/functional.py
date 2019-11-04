import torch.nn.functional as F
from dpn.constants import EPS
from operator import and_
from functools import reduce
from torch.distributions import Dirichlet as Dir


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


        # Sanity checking - at least one is defined.
        # flags = [x is None for x in [logits, probs, alphas]]
        # assert(reduce(and_, flags))

        self.dimB, self.dimH = self.logits.size()

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
    def concentration(self):
        if self._concentration is None:
            self._concentration =  self.alphas.sum(
                dim=self.dimH, 
                keepdim=True
            )
        return self._concentration

    def differential_entropy(self): 
        distribution = Dir(self.alphas)
        return distribution.entropy()

    def expected_entropy(self): 
        alphas = self.alphas
        conc = self.concentration
        expected_entropy = -1 * torch.sum(
                torch.exp(alphas.log() - conc.log())
                * (torch.digamma(alphas + 1.0) - digamma(conc + 1.0))
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
        # Weird. Must numerically verify.
        conc = self.concentration
        alphas = self.alphas

        return -1*torch.sum(
            alphas/conc
            * torch.log(alphas/conc)
            - torch.digamma(alphas + 1.0)
            + torch.digamma(conc   + 1.0)
            ,dim = self.dimH
        )


