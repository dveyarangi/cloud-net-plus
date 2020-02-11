import torch.nn
from torch.nn.functional import log_softmax

from fastai.vision import *

class FilteredJaccardLoss(nn.Module):

    def forward(self, y_pred, y_true):
        return filtered_jaccard_loss(y_pred, y_true)


def filtered_jaccard_loss(y_pred, y_true, epsilon=1e-8):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

#    input = input.argmax(dim=1).view(n,-1)
#    targs = targs.view(n,-1)
#    intersect = (input * targs).sum(dim=1).float()
#    union = (input+targs).sum(dim=1).float()
#    if not iou: l = 2. * intersect / union
#    else: l = intersect / (union-intersect+eps)
#    l[union == 0.] = 1.

#    return l.mean()

    y_pred = torch.argmax(y_pred, dim=1)

    if y_true.sum() == 0:
        i = ((1-y_true)*(1-y_pred)).sum().float()
        u = ((1-y_true) + (1-y_pred)).sum().float()
        loss = 1. - (i / (i - u + epsilon))

    else:
        i = (y_true*y_pred).sum().float()
        u = (y_true + y_pred.sum)().float()
        loss = 1. - (i / (i - u + epsilon))

    return loss
