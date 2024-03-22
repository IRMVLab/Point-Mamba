
import torch
import ocnn
import dwconv

from ocnn.octree import Octree
from typing import Optional, List
from torch.utils.checkpoint import checkpoint

import torch.nn as nn
import torch.nn.functional as F

#DropPath
from timm.models.layers import DropPath

from torch import Tensor
from typing import Optional

#Mamba
import math
from functools import partial

from collections import namedtuple

from mamba_ssm.modules.mamba_simple import Mamba, Block
# from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class OctreeT(Octree):

  def __init__(self, octree: Octree, patch_size: int = 24, dilation: int = 4,
               nempty: bool = True, max_depth: Optional[int] = None,
               start_depth: Optional[int] = None, **kwargs):
    super().__init__(octree.depth, octree.full_depth)
    self.__dict__.update(octree.__dict__)

    # self.patch_size = patch_size
    # self.dilation = dilation  # TODO dilation as a list
    self.nempty = nempty
    self.max_depth = max_depth or self.depth
    self.start_depth = start_depth or self.full_depth
    self.invalid_mask_value = -1e3
    assert self.start_depth > 1

    self.block_num = patch_size * dilation
    self.nnum_t = self.nnum_nempty if nempty else self.nnum
    self.nnum_a = ((self.nnum_t / self.block_num).ceil() * self.block_num).int()

    num = self.max_depth + 1
    self.batch_idx = [None] * num
    self.build_t()

  def build_t(self):
    for d in range(self.start_depth, self.max_depth + 1):
      self.build_batch_idx(d)

  def build_batch_idx(self, depth: int):
    batch = self.batch_id(depth, self.nempty)
    self.batch_idx[depth] = self.patch_partition(batch, depth, self.batch_size)

  def patch_partition(self, data: torch.Tensor, depth: int, fill_value=0):
    num = self.nnum_a[depth] - self.nnum_t[depth]
    tail = data.new_full((num,) + data.shape[1:], fill_value)




class MLP(torch.nn.Module):

  def __init__(self, in_features: int, hidden_features: Optional[int] = None,
               out_features: Optional[int] = None, activation=torch.nn.GELU,
               drop: float = 0.0, **kwargs):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features or in_features
    self.hidden_features = hidden_features or in_features

    self.fc1 = torch.nn.Linear(self.in_features, self.hidden_features)
    self.act = activation()
    self.fc2 = torch.nn.Linear(self.hidden_features, self.out_features)
    self.drop = torch.nn.Dropout(drop, inplace=True)

  def forward(self, data: torch.Tensor):
    data = self.fc1(data)
    data = self.act(data)
    data = self.drop(data)
    data = self.fc2(data)
    data = self.drop(data)
    return data


class OctreeDWConvBn(torch.nn.Module):

  def __init__(self, in_channels: int, kernel_size: List[int] = [3],
               stride: int = 1, nempty: bool = False):
    super().__init__()
    self.conv = dwconv.OctreeDWConv(
        in_channels, kernel_size, nempty, use_bias=False)
    self.bn = torch.nn.BatchNorm1d(in_channels)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    out = self.conv(data, octree, depth)
    out = self.bn(out)
    return out


class RPE(torch.nn.Module):

  def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
    super().__init__()
    self.patch_size = patch_size
    self.num_heads = num_heads
    self.dilation = dilation
    self.pos_bnd = self.get_pos_bnd(patch_size)
    self.rpe_num = 2 * self.pos_bnd + 1
    self.rpe_table = torch.nn.Parameter(torch.zeros(3*self.rpe_num, num_heads))
    torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

  def get_pos_bnd(self, patch_size: int):
    return int(0.8 * patch_size * self.dilation**0.5)

  def xyz2idx(self, xyz: torch.Tensor):
    mul = torch.arange(3, device=xyz.device) * self.rpe_num
    xyz = xyz.clamp(-self.pos_bnd, self.pos_bnd)
    idx = xyz + (self.pos_bnd + mul)
    return idx

  def forward(self, xyz):
    idx = self.xyz2idx(xyz)
    out = self.rpe_table.index_select(0, idx.reshape(-1))
    out = out.view(idx.shape + (-1,)).sum(3)
    out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
    return out

  def extra_repr(self) -> str:
    return 'num_heads={}, pos_bnd={}, dilation={}'.format(
            self.num_heads, self.pos_bnd, self.dilation)  # noqa


  def extra_repr(self) -> str:
    return 'dim={}, patch_size={}, num_heads={}, dilation={}'.format(
            self.dim)  # noqa

#OctMamba
class OctreeMamba(nn.Module):
  
  def __init__(self, dim: int,
                proj_drop: float = 0.0,):
    super().__init__()
    self.dim = dim
    
    self.pim = PointMambaMix(input_dim=dim, output_dim=dim,fused_add_norm=True)
    self.proj = torch.nn.Linear(dim, dim)
    self.proj_drop = torch.nn.Dropout(proj_drop)

  def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
    data = data.unsqueeze(0)
    # data = data.view(-1, K, C)#N,K,C->256,32,256 
    
    # data = data.permute(1, 0, 2)#B,N,C
    data = self.pim(data)
    data = data.squeeze(0)
    data = self.proj(data)
    data = self.proj_drop(data)
    return data
  
  def extra_repr(self) -> str:
    return 'dim={}'.format(
            self.dim)  # noqa


class PointMambaBlock(torch.nn.Module):

  def __init__(self, dim: int,
               proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
               activation: torch.nn.Module = torch.nn.GELU, **kwargs):
    super().__init__()
    self.norm1 = torch.nn.LayerNorm(dim)
    self.mamba = OctreeMamba(dim,proj_drop)
    self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
    self.cpe = OctreeDWConvBn(dim, nempty=nempty)

  def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
    data = self.cpe(data, octree, depth) + data
    attn = self.mamba(self.norm1(data), octree, depth)
    data = data + self.drop_path(attn, octree, depth)
    return data


class PointMambaStage(torch.nn.Module):

  def __init__(self, dim: int, 
               proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
               activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
               use_checkpoint: bool = True, num_blocks: int = 2,
               pim_block=PointMambaBlock, **kwargs):
    super().__init__()
    self.num_blocks = num_blocks
    self.use_checkpoint = use_checkpoint
    self.interval = interval  # normalization interval
    self.num_norms = (num_blocks - 1) // self.interval

    self.blocks = torch.nn.ModuleList([pim_block(
        dim=dim
        , proj_drop=proj_drop,
        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        nempty=nempty, activation=activation) for i in range(num_blocks)])
    # self.norms = torch.nn.ModuleList([
    #     torch.nn.BatchNorm1d(dim) for _ in range(self.num_norms)])

  def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
    for i in range(self.num_blocks):
      if self.use_checkpoint and self.training:
        data = checkpoint(self.blocks[i], data, octree, depth)
      else:
        data = self.blocks[i](data, octree, depth)
      # if i % self.interval == 0 and i != 0:
      #   data = self.norms[(i - 1) // self.interval](data)
    return data


class PatchEmbed(torch.nn.Module):

  def __init__(self, in_channels: int = 3, dim: int = 96, num_down: int = 2,
               nempty: bool = True, **kwargs):
    super().__init__()
    self.num_stages = num_down
    self.delta_depth = -num_down
    channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]

    self.convs = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
        in_channels if i == 0 else channels[i], channels[i], kernel_size=[3],
        stride=1, nempty=nempty) for i in range(self.num_stages)])
    self.downsamples = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
        channels[i], channels[i+1], kernel_size=[2], stride=2, nempty=nempty)
        for i in range(self.num_stages)])
    self.proj = ocnn.modules.OctreeConvBnRelu(
        channels[-1], dim, kernel_size=[3], stride=1, nempty=nempty)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    for i in range(self.num_stages):
      depth_i = depth - i
      data = self.convs[i](data, octree, depth_i)
      data = self.downsamples[i](data, octree, depth_i)
    data = self.proj(data, octree, depth_i - 1)
    return data


class Downsample(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [2], nempty: bool = True):
    super().__init__()
    self.norm = torch.nn.BatchNorm1d(out_channels)
    self.conv = ocnn.nn.OctreeConv(in_channels, out_channels, kernel_size,
                                   stride=2, nempty=nempty, use_bias=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    data = self.conv(data, octree, depth)
    data = self.norm(data)
    return data


class PointMamba(torch.nn.Module):

  def __init__(self, in_channels: int,
               channels: List[int] = [96, 192, 384, 384],
               num_blocks: List[int] = [2, 2, 18, 2],
              #  num_heads: List[int] = [6, 12, 24, 24],
               drop_path: float = 0.5,
               nempty: bool = True, stem_down: int = 2, **kwargs):
    super().__init__()
    self.nempty = nempty
    self.num_stages = len(num_blocks)
    self.stem_down = stem_down
    drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

    self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
    self.layers = torch.nn.ModuleList([PointMambaStage(
        dim=channels[i],
        drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i+1])],
        nempty=nempty, num_blocks=num_blocks[i],)
        for i in range(self.num_stages)])
    self.downsamples = torch.nn.ModuleList([Downsample(
        channels[i], channels[i + 1], kernel_size=[2],
        nempty=nempty) for i in range(self.num_stages - 1)])

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    data = self.patch_embed(data, octree, depth)
    depth = depth - self.stem_down   # current octree depth
    # octree = OctreeT(octree, self.patch_size, self.dilation, self.nempty,
    #                  max_depth=depth, start_depth=depth-self.num_stages+1)
    octree = OctreeT(octree, self.nempty,
                     max_depth=depth, start_depth=depth-self.num_stages+1)
    features = {}
    for i in range(self.num_stages):
      depth_i = depth - i
      data = self.layers[i](data, octree, depth_i)
      features[depth_i] = data
      if i < self.num_stages - 1:
        data = self.downsamples[i](data, octree, depth_i)
    return features



## PointMamba
class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.0
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)



def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, **ssm_cfg, **factory_kwargs)#创建Mamba类的实例
    
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
 
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:

                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
                    

class PointMambaMix(nn.Module):
    def __init__(self,  
        output_dim=512,
        input_dim=512,
        drop_path = 0.1,
        drop_out_in_block= 0.1,
        n_layer=1,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        # bimamba_type="none",
        bimamba_type="v2",
        **kwargs)->None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.input_dim = input_dim
        self.output_dim = output_dim 
        
        
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    input_dim,#d_model
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            input_dim, eps=norm_epsilon, **factory_kwargs
        )
        
        self.pre_logits = nn.Identity()
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
        

    def forward_features(self, input_ids, inference_params=None):
        # hidden_states = self.embedding(input_ids)
        hidden_states = input_ids
        
        # print('input_ids.shape',input_ids.shape)
        residual = None
        
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn

            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
            

        #offset
        hidden_states = hidden_states - input_ids

        return hidden_states
    
    def forward(self, input_ids, inference_params=None):
        return input_ids