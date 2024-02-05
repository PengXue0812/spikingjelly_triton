import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import surrogate, functional

class SpearablePSN(nn.Module): 
    def __init__(self, T: int, P: int, surrogate_function=surrogate.ATan()):
        super().__init__()
        self.T = T
        self.P = P
        self.wa = nn.Parameter(torch.zeros([T, P]))
        self.wb = nn.Parameter(torch.zeros([P, T]))

        lam = math.sqrt(3 / math.sqrt(3 * T * P))
        nn.init.uniform_(self.wa, -lam, lam)
        nn.init.uniform_(self.wb, -lam, lam)

        self.bias = nn.Parameter(- torch.ones([T, 1]))
        self.surrogate_function = surrogate_function

    def forward(self, x_seq: torch.Tensor, h_bias=None):
        # x_seq.shape = [T, N]
        weight = self.wa @ self.wb
        bias = self.bias

        h_seq = torch.addmm(bias, weight, x_seq.flatten(1)).view(x_seq.shape)
        if h_bias is not None:
            h_seq = h_seq + h_bias
        spike_seq = self.surrogate_function(h_seq)
        return spike_seq

def conv_bn_infer_forward(x_seq: torch.Tensor, conv: nn.Conv2d, bn: nn.BatchNorm1d or None, spearable_psn: SpearablePSN, pre_unbiased_layers: nn.Module = None):
    if bn is not None:
        assert conv.bias is None

    T = spearable_psn.T
    P = spearable_psn.P

    # bias.shape = [C]
    if bn is not None:
        bias = functional.fused_conv2d_bias_of_convbn2d(conv, bn)
    else:
        bias = conv.bias
    C = bias.shape[0]

    # spearable_psn.wb @ bias: [P, T] @ [T, C] = [P, C]
    offset = bias - spearable_psn.wb @ bias.view(1, C).repeat(T, 1)
    # offset.shape = [P, C]

    # spearable_psn.wa @ offset: [T, P]  @ [P, C] = [T, C]
    offset = spearable_psn.wa @ offset

    offset = offset.view(T, 1, C, 1, 1)
    
    mds = []
    if pre_unbiased_layers is not None:
        mds.append(pre_unbiased_layers)
    mds.append(conv)
    if bn is not None:
        mds.append(bn)
    return spearable_psn(functional.seq_to_ann_forward(x_seq, mds), offset)


def conv_bn_train_forward(x_seq: torch.Tensor, conv: nn.Conv2d, bn: nn.BatchNorm1d or None, spearable_psn: SpearablePSN, pre_unbiased_layers: nn.Module = None):
    if bn is not None:
        assert conv.bias is None

    T = spearable_psn.T
    P = spearable_psn.P

    y_seq = spearable_psn.wb @ x_seq.flatten(1)
    y_seq = y_seq.view([P, ] + list(x_seq.shape[1:]))
    
    mds = []
    if pre_unbiased_layers is not None:
        mds.append(pre_unbiased_layers)
    mds.append(conv)
    if bn is not None:
        mds.append(bn)
        

    y_seq = functional.seq_to_ann_forward(y_seq, mds)

    h_seq = torch.addmm(spearable_psn.bias, spearable_psn.wa, y_seq.flatten(1)).view([T, ] + list(y_seq.shape[1:]))
    spike_seq = spearable_psn.surrogate_function(h_seq)
    return spike_seq

def conv_bn_forward(x_seq: torch.Tensor, conv: nn.Conv2d, bn: nn.BatchNorm1d or None, spearable_psn: SpearablePSN, pre_unbiased_layers: nn.Module = None):
    if spearable_psn.training:
        return conv_bn_train_forward(x_seq, conv, bn, spearable_psn, pre_unbiased_layers)
    else:
        return conv_bn_infer_forward(x_seq, conv, bn, spearable_psn, pre_unbiased_layers)

def linear_train_forward(x_seq: torch.Tensor, fc: nn.Linear, spearable_psn: SpearablePSN, pre_unbiased_layers: nn.Module = None):
    T = spearable_psn.T
    P = spearable_psn.P

    y_seq = spearable_psn.wb @ x_seq.flatten(1)
    y_seq = y_seq.view([P, ] + list(x_seq.shape[1:]))
    mds = []
    if pre_unbiased_layers is not None:
        mds.append(pre_unbiased_layers)
    mds.append(fc)
    y_seq = functional.seq_to_ann_forward(y_seq, mds)
    h_seq = torch.addmm(spearable_psn.bias, spearable_psn.wa, y_seq.flatten(1)).view([T, ] + list(y_seq.shape[1:]))
    spike_seq = spearable_psn.surrogate_function(h_seq)
    return spike_seq


def linear_infer_forward(x_seq: torch.Tensor, fc: nn.Linear, spearable_psn: SpearablePSN, pre_unbiased_layers: nn.Module = None):
    if fc.bias is not None:
        T = spearable_psn.T
        P = spearable_psn.P

        # bias.shape = [C]
        bias = fc.bias
        C = bias.shape[0]

        # spearable_psn.wb @ bias: [P, T] @ [T, C] = [P, C]
        offset = bias - spearable_psn.wb @ bias.view(1, C).repeat(T, 1)
        # offset.shape = [P, C]

        # spearable_psn.wa @ offset: [T, P]  @ [P, C] = [T, C]
        offset = spearable_psn.wa @ offset

        offset = offset.view(T, 1, C)

        mds = []
        if pre_unbiased_layers is not None:
            mds.append(pre_unbiased_layers)
        mds.append(fc)

        return spearable_psn(functional.seq_to_ann_forward(x_seq, mds), offset)
    else:
        return spearable_psn(functional.seq_to_ann_forward(x_seq, fc))


def linear_forward(x_seq: torch.Tensor, fc: nn.Linear, spearable_psn: SpearablePSN, pre_unbiased_layers: nn.Module = None):
    if spearable_psn.training:
        return linear_train_forward(x_seq, fc, spearable_psn, pre_unbiased_layers)
    else:
        return linear_infer_forward(x_seq, fc, spearable_psn, pre_unbiased_layers)


def general_train_forward(x_seq: torch.Tensor, linear_layers: nn.Module or tuple, spearable_psn: SpearablePSN):
    T = spearable_psn.T
    P = spearable_psn.P

    y_seq = spearable_psn.wb @ x_seq.flatten(1)
    y_seq = y_seq.view([P, ] + list(x_seq.shape[1:]))
    y_seq = functional.seq_to_ann_forward(y_seq, linear_layers)
    h_seq = torch.addmm(spearable_psn.bias, spearable_psn.wa, y_seq.flatten(1)).view([T, ] + list(y_seq.shape[1:]))
    spike_seq = spearable_psn.surrogate_function(h_seq)
    return spike_seq

def general_infer_forward(x_seq: torch.Tensor, linear_layers: nn.Module or tuple, spearable_psn: SpearablePSN):
    T = spearable_psn.T
    P = spearable_psn.P

    bias = functional.seq_to_ann_forward(torch.zeros([1, 1, ] + list(x_seq.shape[2:]), device=x_seq.device, dtype=x_seq.dtype), linear_layers)

    # bias.shape = [C, ?]

    # spearable_psn.wb @ bias: [P, T] @ [T, C * ?] = [P, C * ?]
    offset = bias.flatten() - spearable_psn.wb @ bias.view(1, -1).repeat(T, 1)
    # offset.shape = [P, C * ?]

    # spearable_psn.wa @ offset: [T, P]  @ [P, C * ?] = [T, C * ?]
    offset = spearable_psn.wa @ offset

    offset = offset.view([T, 1] + list(bias.shape[2:]))

    return spearable_psn(functional.seq_to_ann_forward(x_seq, linear_layers), offset)

def general_forward(x_seq: torch.Tensor, linear_layers: nn.Module or tuple, spearable_psn: SpearablePSN):
    if spearable_psn.training:
        return general_train_forward(x_seq, linear_layers, spearable_psn)
    else:
        return general_infer_forward(x_seq, linear_layers, spearable_psn)
