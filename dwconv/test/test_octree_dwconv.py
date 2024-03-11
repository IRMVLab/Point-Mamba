import os
import torch
import ocnn
import unittest
import dwconv
from dwconv.core import dwconv_forward_backward, dwconv_weight_backward, inverse_neigh

from .utils import get_batch_octree


class TesOctreeDWConv(unittest.TestCase):

  def test_dwconv_with_conv(self):

    depth = 4
    channel = 256
    octree = get_batch_octree().cuda()
    kernel_size = [[3, 3, 3], [3, 1, 1], [1, 3, 1], [1, 1, 3],
                   [2, 2, 2], [3, 3, 1], [1, 3, 3], [3, 1, 3]]

    for i in range(len(kernel_size)):
      for nempty in [True, False]:
        # ocnn.nn.OctreeDWConv
        nnum = octree.nnum_nempty[depth] if nempty else octree.nnum[depth]
        rnd_data = torch.randn(nnum, channel).cuda()
        ocnn_data = rnd_data.clone().requires_grad_()
        ocnn_conv = ocnn.nn.OctreeDWConv(channel, kernel_size[i], nempty=nempty)
        ocnn_conv.cuda()
        ocnn_out = ocnn_conv(ocnn_data, octree, depth)
        ocnn_out.sum().backward()

        # prepare test data
        data = rnd_data.clone().requires_grad_()
        weights = ocnn_conv.weights.detach().clone().requires_grad_()
        kernel = ''.join([str(k) for k in kernel_size[i]])
        neigh = octree.get_neigh(depth, kernel, nempty=nempty)

        # test the api seperatly
        out = dwconv_forward_backward(data, weights, neigh)
        grad = torch.full_like(data, fill_value=1)
        ineigh = inverse_neigh(neigh)
        grad_d = dwconv_forward_backward(grad, weights, ineigh)
        grad_w = dwconv_weight_backward(grad, data, neigh)
        self.assertTrue(torch.allclose(out, ocnn_out, atol=1e-6))
        self.assertTrue(torch.allclose(grad_d, ocnn_data.grad, atol=1e-6))
        self.assertTrue(torch.allclose(
            grad_w, ocnn_conv.weights.grad, atol=5e-5))

        # test the autograd function
        out = dwconv.octree_dwconv(data, weights, neigh)
        out.sum().backward()
        self.assertTrue(torch.allclose(out, ocnn_out, atol=1e-6))
        self.assertTrue(torch.allclose(data.grad, ocnn_data.grad, atol=1e-6))
        self.assertTrue(torch.allclose(
            weights.grad, ocnn_conv.weights.grad, atol=5e-5))

        # test the module
        data = rnd_data.clone().requires_grad_()
        octree_dwconv = dwconv.OctreeDWConv(channel, kernel_size[i], nempty)
        octree_dwconv.cuda()
        octree_dwconv.weights.data.copy_(ocnn_conv.weights.data)
        out = octree_dwconv(data, octree, depth)
        out.sum().backward()

        self.assertTrue(torch.allclose(out, ocnn_out, atol=1e-6))
        self.assertTrue(torch.allclose(data.grad, ocnn_data.grad, atol=1e-6))
        self.assertTrue(torch.allclose(
            octree_dwconv.weights.grad, ocnn_conv.weights.grad, atol=5e-5))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
