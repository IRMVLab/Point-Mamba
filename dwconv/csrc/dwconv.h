// --------------------------------------------------------
// Octree-based Sparse Convolutional Neural Networks
// Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
// Licensed under The MIT License [see LICENSE for details]
// Written by Peng-Shuai Wang
// --------------------------------------------------------

#pragma once
#include <torch/extension.h>

using torch::Tensor;

Tensor dwconv_forward_backward(Tensor data, Tensor weight, Tensor neigh);
Tensor dwconv_weight_backward(Tensor grad, Tensor data, Tensor neigh);
Tensor inverse_neigh(Tensor neigh);
