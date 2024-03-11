// --------------------------------------------------------
// Octree-based Sparse Convolutional Neural Networks
// Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
// Licensed under The MIT License [see LICENSE for details]
// Written by Peng-Shuai Wang
// --------------------------------------------------------

#include "dwconv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dwconv_forward_backward", &dwconv_forward_backward, "forward_backward");
  m.def("dwconv_weight_backward", &dwconv_weight_backward, "weight_backward");
  m.def("inverse_neigh", &inverse_neigh, "inverse_neigh");
}
