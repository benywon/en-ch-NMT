# -*- coding: utf-8 -*-
"""
 @Time    : 2018/7/19 下午5:46
 @FileName: torchUtils.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

def get_tensor_data(tensor, is_gpu=True):
    if is_gpu:
        return tensor.data.cpu().numpy()
    return tensor.data.numpy()


def get_model_parameters(model):
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            tmp = 1
            for a in parameter.size():
                tmp *= a
            total += tmp
    return total


def tensordot_pytorch(a, b, axes=2):
    # code adapted from numpy
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    # uncomment in pytorch >= 0.5
    # a, b = torch.as_tensor(a), torch.as_tensor(b)
    as_ = a.shape
    nda = a.dim()
    bs = b.shape
    ndb = b.dim()
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.permute(newaxes_a).reshape(newshape_a)
    bt = b.permute(newaxes_b).reshape(newshape_b)

    res = at.matmul(bt)
    return res.reshape(olda + oldb)


def cosine(a, b):
    num = a.mm(b.transpose(0, 1))
    dom = a.norm(dim=1).unsqueeze(1).mm(b.norm(dim=1).unsqueeze(0)) + 1.0e-8
    return num / dom


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def get_att_output(tensor, att_module):
    value = att_module(tensor)
    score = F.softmax(value, 1)
    output = score.transpose(2, 1).bmm(tensor)
    return output


def get_model(filename):
    with open(filename, 'rb') as f:
        model = torch.load(f,map_location='cpu')
    return model
