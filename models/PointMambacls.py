
import ocnn
import torch

from ocnn.octree import Octree
from typing import List

from .PointMamba import PointMamba


class ClsHeader(torch.nn.Module):
  def __init__(self, out_channels: int, in_channels: int,
               nempty: bool = False, dropout: float = 0.5):
    super().__init__()
    self.global_pool = ocnn.nn.OctreeGlobalPool(nempty)
    self.cls_header = torch.nn.Sequential(
        ocnn.modules.FcBnRelu(in_channels, 256),
        torch.nn.Dropout(p=dropout),
        torch.nn.Linear(256, out_channels))

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    data = self.global_pool(data, octree, depth)
    logit = self.cls_header(data)
    return logit


class PointMambaCls(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int,
               channels: List[int] = [96, 192, 384, 384],
               num_blocks: List[int] = [2, 2, 18, 2],
               drop_path: float = 0.5, nempty: bool = True,
               stem_down: int = 2, head_drop: float = 0.5, **kwargs):
    super().__init__()
    self.backbone = PointMamba(
        in_channels, channels, num_blocks,
        drop_path, nempty, stem_down)
    self.head = ClsHeader(
        out_channels, channels[-1], nempty, head_drop)
    self.apply(self.init_weights)

  def init_weights(self, m):
    if isinstance(m, torch.nn.Linear):
      torch.nn.init.trunc_normal_(m.weight, std=0.02)
      if isinstance(m, torch.nn.Linear) and m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    features = self.backbone(data, octree, depth)
    curr_depth = min(features.keys())
    output = self.head(features[curr_depth], octree, curr_depth)
    return output
