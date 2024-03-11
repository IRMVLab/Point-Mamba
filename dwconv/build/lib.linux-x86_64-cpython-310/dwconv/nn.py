# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import ocnn
from typing import List
from ocnn.octree import Octree
from torch.autograd import Function

from .core import dwconv_forward_backward, dwconv_weight_backward, inverse_neigh


class OctreeDWConvFunction(Function):
  r''' Wrap the octree depth-wise convolution with auto-diff.
  '''

  @staticmethod
  def forward(ctx, data: torch.Tensor, weights: torch.Tensor, neigh: torch.Tensor):
    data = data.contiguous()
    weights = weights.contiguous()
    neigh = neigh.contiguous()
    out = dwconv_forward_backward(data, weights, neigh)
    ctx.save_for_backward(data, weights, neigh)
    return out

  @staticmethod
  def backward(ctx, grad):
    data, weights, neigh = ctx.saved_tensors
    grad = grad.contiguous()

    grad_d = None
    if ctx.needs_input_grad[0]:
      ineigh = inverse_neigh(neigh)
      grad_d = dwconv_forward_backward(grad, weights, ineigh)

    grad_w = None
    if ctx.needs_input_grad[1]:
      grad_w = dwconv_weight_backward(grad, data, neigh)
    return grad_d, grad_w, None


octree_dwconv = OctreeDWConvFunction.apply


class OctreeDWConv(ocnn.nn.OctreeDWConv):
  r''' Speeds up `ocnn.nn.OctreeDWConv` with CUDA.
  '''

  def __init__(self, channels: int, kernel_size: List[int] = [3],
               nempty: bool = False, use_bias: bool = False):
    super().__init__(in_channels=channels, kernel_size=kernel_size, stride=1,
                     nempty=nempty, use_bias=use_bias)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    neigh = octree.get_neigh(depth, self.kernel, self.stride, self.nempty)
    out = octree_dwconv(data, self.weights, neigh)
    if self.use_bias:
      out += self.bias
    return out
