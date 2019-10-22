

from torch import nn
import torch 
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

class ExpectedKL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mean, precision, true_mean, true_precision):

        # cost = T(alpha)/T(precision)
        # cost = tf.lgamma(target_precision + epsilon) - tf.lgamma(precision + epsilon) \
        #        + tf.reduce_sum(
        #     tf.lgamma(mean * precision + epsilon) - tf.lgamma(target_mean * target_precision + epsilon), axis=1) \
        #        + tf.reduce_sum((target_precision * target_mean - mean * precision) * (
        # tf.digamma(target_mean * target_precision + epsilon) -
        # tf.digamma(target_precision + epsilon)), axis=1)
        # cost = tf.reduce_mean(cost)

        # Is there some Log added for normal distributions?
        dirichlet = Dirichlet(precision)

        pass



