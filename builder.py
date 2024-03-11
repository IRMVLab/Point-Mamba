
import ocnn
import torch
import datasets
import models

import os

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'  # 或者 'DETAIL'


def PointMambaSeg_large(in_channels, out_channels, **kwargs):
  return models.PointMambaSeg(
      in_channels, out_channels,
      channels=[192, 384, 768, 768],
      num_blocks=[2, 2, 18, 2],
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def PointMambaSeg_base(in_channels, out_channels, **kwargs):
  return models.PointMambaSeg(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[2, 2, 18, 2],
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def PointMambaSeg_small(in_channels, out_channels, **kwargs):
  return models.PointMambaSeg(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[2, 2, 6, 2],
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def PointMamba_cls(in_channels, out_channels, nemtpy, **kwargs):
  return models.PointMambaCls(
      in_channels, out_channels,
      channels=[96, 192],
      num_blocks=[6, 6],
      drop_path=0.5, nempty=nemtpy,
      stem_down=2, head_drop=0.5)


def get_segmentation_model(flags):
  params = {
      'in_channels': flags.channel, 'out_channels': flags.nout,
      'interp': flags.interp, 'nempty': flags.nempty,
  }
  networks = {
      # 'octsegformer': octsegformer,
      # 'octsegformer_large': octsegformer_large,
      # 'octsegformer_small': octsegformer_small,
      'pointmamba_seg': PointMambaSeg_base,
      'pointmamba_seg_large': PointMambaSeg_large,
      'pointmamba_seg_small': PointMambaSeg_small,
  }

  return networks[flags.name.lower()](**params)#.lower()表示将字符串转换为小写


def get_classification_model(flags):
  if flags.name.lower() == 'lenet':
    model = ocnn.models.LeNet(
        flags.channel, flags.nout, flags.stages, flags.nempty)
  elif flags.name.lower() == 'hrnet':
    model = ocnn.models.HRNet(
        flags.channel, flags.nout, flags.stages, nempty=flags.nempty)
  elif flags.name.lower() == 'pointmamba_cls':
    model = PointMamba_cls(flags.channel, flags.nout, flags.nempty)
  else:
    raise ValueError
  return model


def get_segmentation_dataset(flags):
  if flags.name.lower() == 'shapenet':
    return datasets.get_shapenet_seg_dataset(flags)
  elif flags.name.lower() == 'scannet':
    return datasets.get_scannet_dataset(flags)
  elif flags.name.lower() == 'kitti':
    return datasets.get_kitti_dataset(flags)
  else:
    raise ValueError
