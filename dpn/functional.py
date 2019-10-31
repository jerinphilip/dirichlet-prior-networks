import torch.nn.functional as F


def entropy_from_logits(logits):
    dimB, dimH = 0, 1
    probs = F.softmax(logits, dim=dimH)
    logprobs = F.log_softmax(logits, dim=dimH)
    plogp = probs*logprobs
    entropy = -1*plogp.sum(dim=dimH)
    return entropy
