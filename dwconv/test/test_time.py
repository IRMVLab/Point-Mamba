import os
import torch
import ocnn
import dwconv
import time
import numpy as np

pts = torch.from_numpy(np.load('test/data/pts.npy'))
points = ocnn.octree.Points(points=pts[:, :3], features=pts[:, 3:])

# octree
depth = 10
nempty = True
octree = ocnn.octree.Octree(depth=depth)
octree.build_octree(points)
octree = octree.cuda()
octree.construct_all_neigh()

# data
channel = 96
nnum = octree.nnum_nempty[depth] if nempty else octree.nnum[depth]
rnd_data = torch.randn(nnum, channel).cuda()
ocnn_data = rnd_data.clone().requires_grad_()
dw_data = rnd_data.clone().requires_grad_()

# ocnn
ocnn_time = []
ocnn_conv = ocnn.nn.OctreeDWConv(channel, kernel_size=[3], nempty=nempty).cuda()
for i in range(10):
  t1 = time.perf_counter()
  ocnn_out = ocnn_conv(ocnn_data, octree, depth)
  ocnn_out.sum().backward()
  torch.cuda.synchronize()
  ocnn_time.append(time.perf_counter() - t1)


# test the module
dw_time = []
dw_conv = dwconv.OctreeDWConv(channel, kernel_size=[3], nempty=nempty).cuda()
dw_conv.weights.data.copy_(ocnn_conv.weights.data)
for i in range(10):
  t1 = time.perf_counter()
  out = dw_conv(dw_data, octree, depth)
  out.sum().backward()
  torch.cuda.synchronize()
  dw_time.append(time.perf_counter() - t1)

print('out allclose: ', torch.allclose(out, ocnn_out, atol=1.0e-7))
print('out grad allclose: ',
      torch.allclose(dw_data.grad, ocnn_data.grad, atol=1.0e-6, rtol=1.0e-6))
print('weigth grad allclose: ',
      torch.allclose(dw_conv.weights.grad, ocnn_conv.weights.grad, rtol=9.0e-4))

print('ocnn time: ', ocnn_time[5:])
print('dw time: ', dw_time[5:])
