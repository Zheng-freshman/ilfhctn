# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.layers import DropPath
from timm.layers.helpers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


from math import pi, log
from functools import wraps
from typing import *

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce


# HELPERS/UTILS
"""
Helper class implementations based on: https://github.com/lucidrains/perceiver-pytorch
"""


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class SELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.selu(gates)

class RELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.relu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., snn: bool = False):
        super().__init__()
        activation = SELU() if snn else GELU()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            activation,
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def temperature_softmax(logits, temperature=1.0, dim=-1):
    """
    Temperature scaled softmax
    Args:
        logits:
        temperature:
        dim:

    Returns:
    """
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=dim)



class ContextAttention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        # add leaky relu
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.LeakyReLU(negative_slope=1e-2)
        )

        self.attn_weights = None
        # self._init_weights()

    def _init_weights(self):
    # Use He initialization for Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # Initialize bias to zero if there's any
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        # attn = sim.softmax(dim = -1)
        attn = temperature_softmax(sim, temperature=0.5, dim=-1)
        self.attn_weights = attn
        attn = self.dropout(attn)


        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)


class FusionNet(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, *,
            n_modalities: int,channel_dims: List,num_spatial_axes: List, out_dims: int,
            context_depth: int = 3, num_freq_bands: int = 2, max_freq: float=10.,
            l_c: int = 1, l_d: int = 384, x_heads: int = 8, l_heads: int = 8,
            cross_dim_head: int = 64, latent_dim_head: int = 64,
            attn_dropout: float = 0., ff_dropout: float = 0.,
            weight_tie_layers: bool = False, fourier_encode_data: bool = True,
            self_per_cross_attn: int = 1,
            final_classifier_head: bool = True, snn: bool = True,
            fusing: bool = True, threeDim: bool = False,
            img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, backbone_depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block,):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            backbone_depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init: (str): weight init scheme
            init_values: (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()

        ########## HealNet Block ##########
        assert len(channel_dims) == len(num_spatial_axes), 'input channels and input axis must be of the same length'
        assert len(num_spatial_axes) == n_modalities, 'input axis must be of the same length as the number of modalities'
        self.input_axes = num_spatial_axes
        self.input_channels=channel_dims
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.modalities = n_modalities
        self.self_per_cross_attn = self_per_cross_attn
        self.fourier_encode_data = fourier_encode_data
        self.context_depth = context_depth
        self.backbone_depth = backbone_depth
        self.fusing = fusing
        self.threeDim = threeDim

        # get fourier channels and input dims for each modality
        fourier_channels = []
        input_dims = []
        for axis in num_spatial_axes:
            fourier_channels.append((axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0)
        for f_channels, i_channels in zip(fourier_channels, channel_dims):
            input_dims.append(f_channels + i_channels)

        # initialise shared latent bottleneck
        self.latents = nn.Parameter(torch.randn(l_c, l_d))

        # modality-specific attention layers
        funcs = []
        for m in range(n_modalities):
            funcs.append(lambda m=m: PreNorm(l_d, ContextAttention(l_d, input_dims[m], heads = x_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dims[m]))
        cross_attn_funcs = tuple(map(cache_fn, tuple(funcs)))

        get_latent_attn = lambda: PreNorm(l_d, ContextAttention(l_d, heads = l_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_cross_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout = ff_dropout, snn = snn))
        get_latent_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout = ff_dropout, snn = snn))

        get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(context_depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(get_latent_attn(**cache_args, key = block_ind))
                self_attns.append(get_latent_ff(**cache_args, key = block_ind))

            cross_attn_layers = []
            for j in range(n_modalities):
                cross_attn_layers.append(cross_attn_funcs[j](**cache_args))
                cross_attn_layers.append(get_cross_ff(**cache_args))

            self.layers.append(nn.ModuleList(
                [*cross_attn_layers, self_attns])
            )

        # self.to_logits = nn.Sequential(
        #     Reduce('b n d -> b d', 'mean'),
        #     nn.LayerNorm(l_d),
        #     nn.Linear(l_d, out_dims)
        # ) if final_classifier_head else nn.Identity()

        ##########  Transformer Block ##########
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, backbone_depth)]  # stochastic depth decay rule
        # self.blocks = nn.Sequential(*[
        #     block_fn(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(backbone_depth)])
        assert backbone_depth%context_depth==0, "backbone_depth should be divide by context_depth"
        layer_block_depth = backbone_depth//context_depth
        self.blocks = nn.ModuleList([ nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j*layer_block_depth+i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(layer_block_depth)])
            for j in range(context_depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.num_features = self.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def context_embedding(self,
                          tensors: List[Union[torch.Tensor, None]],
                          verbose: bool = False,
                          ):
        
        missing_idx = [i for i, t in enumerate(tensors) if t is None]
        if verbose: 
            print(f"Missing modalities indices: {missing_idx}")
        for i in range(len(tensors)):
            if i in missing_idx: 
                continue
            else: 
                data = tensors[i]
                # sanity checks
                b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
                if len(axis)<1: #(b,dim)-->(b,1,dim)
                    axis = [1]
                    data = torch.unsqueeze(data, 1)
                assert len(axis) == self.input_axes[i], (f'input data for modality {i+1} must hav'
                                                            f' the same number of axis as the input axis parameter')

            # fourier encode for each modality
            if self.fourier_encode_data:
                axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
                pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
                enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
                enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
                enc_pos = repeat(enc_pos, '... -> b ...', b = b)
                data = torch.cat((data, enc_pos), dim = -1)
                

            # concat and flatten axis for each modality
            data = rearrange(data, 'b ... d -> b (...) d')
            tensors[i] = data
        
        batch_latents = repeat(self.latents, 'n d -> b n d', b = b)
        return tensors,batch_latents,missing_idx

    def latentAttn(self, feature, latent): #(b,patchs,dim_f),(b,channel,dim_l)
        _, fea_patchs, fea_dim = feature.shape
        _, la_c, la_dim = latent.shape
        output = []
        group = fea_patchs//la_c
        left = fea_patchs%la_c
        for i in range(la_c):
            if i==0:
                begin = i*group  #如果不能整除，余数放第一组
                end = (i+1)*group+left
            else:
                begin = i*group+left
                end = (i+1)*group+left
            if la_dim<fea_dim:
                latent_expend = nn.functional.interpolate(latent[:,i:i+1,:], size=[fea_dim], mode='linear', align_corners=False)
            else:
                latent_expend = latent[:,i:i+1,:]
            output.append(feature[:,begin:end,:]+latent_expend)
        return torch.cat(output,dim=1)

    def extractB(self, x, tensors, mask, verbose):
        tensors_posed, batch_latents, missing_idx = self.context_embedding(tensors+x,verbose)
        # x = self.patch_embed(x)
        # x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        # x = self.pos_drop(x + self.pos_embed)
        for i in range(len(x)):
            x[i] = self.patch_embed(x[i])
            x[i] = torch.cat((self.cls_token.expand(x[i].shape[0], -1, -1), x[i]), dim=1)
            x[i] = self.pos_drop(x[i] + self.pos_embed)

        
        for layer_idx, layer in enumerate(self.layers):
            # x = self.blocks[layer_idx](x)
            for i in range(len(x)):
                x[i] = self.blocks[layer_idx](x[i])
            if self.fusing:
                for i in range(self.modalities):
                    if i in missing_idx: 
                        if verbose: 
                            print(f"Skipping update in fusion layer {layer_idx + 1} for missing modality {i+1}")
                            continue
                    cross_attn=layer[i*2]
                    cross_ff = layer[(i*2)+1]
                    try:
                        batch_latents = cross_attn(batch_latents, context = tensors_posed[i], mask = mask) + batch_latents
                        batch_latents = cross_ff(batch_latents) + batch_latents
                    except:
                        pass

                    if self.self_per_cross_attn > 0:
                        self_attn, self_ff = layer[-1]

                        batch_latents = self_attn(batch_latents) + batch_latents
                        batch_latents = self_ff(batch_latents) + batch_latents
                for i in range(len(x)):
                    x[i] = self.latentAttn(x[i],batch_latents) #(b,patchs,dim) = (b,patchs,dim) + (b,1,dim)
                # x=x+batch_latents


        # x = self.norm(x)
        # return x
        for i in range(len(x)):
            x[i] = self.norm(x[i])
        return torch.stack(x,dim=0).mean(dim=0)

    def extract3D(self, x, tensors, mask, verbose):
        tensors_posed, batch_latents, missing_idx = self.context_embedding(tensors+[torch.cat(x,dim=-3)],verbose)
        for i in range(len(x)):
            x[i] = self.patch_embed(x[i])
            x[i] = torch.cat((self.cls_token.expand(x[i].shape[0], -1, -1), x[i]), dim=1)
            x[i] = self.pos_drop(x[i] + self.pos_embed)

        
        for layer_idx, layer in enumerate(self.layers):
            for i in range(len(x)):
                x[i] = self.blocks[layer_idx](x[i])
            if self.fusing:
                for i in range(self.modalities):
                    if i in missing_idx: 
                        if verbose: 
                            print(f"Skipping update in fusion layer {layer_idx + 1} for missing modality {i+1}")
                            continue
                    cross_attn=layer[i*2]
                    cross_ff = layer[(i*2)+1]
                    try:
                        batch_latents = cross_attn(batch_latents, context = tensors_posed[i], mask = mask) + batch_latents
                        batch_latents = cross_ff(batch_latents) + batch_latents
                    except:
                        pass

                    if self.self_per_cross_attn > 0:
                        self_attn, self_ff = layer[-1]

                        batch_latents = self_attn(batch_latents) + batch_latents
                        batch_latents = self_ff(batch_latents) + batch_latents
                for i in range(len(x)):
                    x[i]=x[i]+batch_latents #(b,patchs,dim) = (b,patchs,dim) + (b,1,dim)

        for i in range(len(x)):
            x[i] = self.norm(x[i])
        return torch.stack(x,dim=0).mean(dim=0)


    def forward(self, x, tensors: List[Union[torch.Tensor, None]],
                only_fc=False, only_feat=False,
                mask: Optional[torch.Tensor] = None,
                return_embeddings: bool = False, 
                verbose: bool = False,
                **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        
        if only_fc:
            return self.head(x)

        if self.threeDim:
            x = self.extract3D(x, tensors, mask, verbose)
        else:
            x = self.extractB(x, tensors, mask, verbose)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)

        if only_feat:
            return x

        output = self.head(x)
        result_dict = {'logits':output, 'feat':x}
        return result_dict

    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        Helper function which returns all attention weights for all attention layers in the model
        Returns:
            all_attn_weights: list of attention weights for each attention layer
        """
        all_attn_weights = []
        for module in self.modules():
            if isinstance(module, ContextAttention):
                if module.attn_weights is not None:
                    all_attn_weights.append(module.attn_weights)
        return all_attn_weights
    
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def group_matcher(self, coarse=False, prefix=''):
        return dict(
            stem=r'^{}cls_token|{}pos_embed|{}patch_embed'.format(prefix, prefix, prefix),  # stem and embed
            blocks=[(r'^{}blocks\.(\d+)'.format(prefix), None), (r'^{}norm'.format(prefix), (99999,))]
        )


def funet_3Dimg_6in1(pretrained=False, pretrained_path=None, num_classes=2, fusing=True, pretrained_from="ViT", context_depth=4,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    embed_dim = 384
    model_kwargs = dict(
        context_depth = context_depth,
        l_c = 1,
        l_d = embed_dim,
        n_modalities = 2,
        channel_dims = [6,60],
        num_spatial_axes = [1,2], 
        out_dims = num_classes,
        num_classes = num_classes,
        fusing = fusing, threeDim = True,
        patch_size=16, embed_dim=embed_dim, backbone_depth=12, num_heads=6, drop_path_rate=0.2,
        **kwargs,
        )
    model = FusionNet(**model_kwargs)
    fromViT = False
    if pretrained_from == "ViT":
        fromViT=True
    if pretrained:
        model = load_checkpoint(model, pretrained_path, model.backbone_depth, model.context_depth,fromViT)
    return model

def funet_2img_52in1(pretrained=False, pretrained_path=None, num_classes=2, fusing=True, pretrained_from="ViT", context_depth=4, l_c=1, l_d=384, **kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    embed_dim = 384
    model_kwargs = dict(
        context_depth = context_depth,
        l_c = l_c,
        l_d = l_d,
        n_modalities = 3,
        channel_dims = [52,3,3],
        num_spatial_axes = [1,2,2], 
        out_dims = num_classes,
        num_classes = num_classes,
        fusing = fusing,
        patch_size=16, embed_dim=embed_dim, backbone_depth=12, num_heads=6, drop_path_rate=0.2, 
        **kwargs
        )
    model = FusionNet(**model_kwargs)
    fromViT = False
    if pretrained_from == "ViT":
        fromViT=True
    if pretrained:
        model = load_checkpoint(model, pretrained_path, model.backbone_depth, model.context_depth,fromViT)
    return model

def funet_1img_52pro(pretrained=False, pretrained_path=None, num_classes=2, fusing=True, pretrained_from="ViT", context_depth=4, **kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    embed_dim = 384
    model_kwargs = dict(
        context_depth = context_depth,
        l_c = 1,
        l_d = embed_dim,
        n_modalities = 53,
        channel_dims = [1 for _ in range(52)]+[3],
        num_spatial_axes = [1 for _ in range(52)]+[2], 
        out_dims = num_classes,
        num_classes = num_classes,
        fusing = fusing,
        patch_size=16, embed_dim=embed_dim, backbone_depth=12, num_heads=6, drop_path_rate=0.2, 
        **kwargs
        )
    model = FusionNet(**model_kwargs)
    fromViT = False
    if pretrained_from == "ViT":
        fromViT=True
    if pretrained:
        model = load_checkpoint(model, pretrained_path, model.backbone_depth, model.context_depth, fromViT)
    return model

def funet_2img_5groups(pretrained=False, pretrained_path=None, num_classes=2, fusing=True, pretrained_from="ViT", context_depth=4,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    embed_dim = 384
    model_kwargs = dict(
        context_depth = context_depth,
        l_c = 1,
        l_d = embed_dim,
        n_modalities = 7,
        channel_dims = [7,13,12,13,7,3,3],
        num_spatial_axes = [1,1,1,1,1,2,2], 
        out_dims = num_classes,
        num_classes = num_classes,
        fusing = fusing,
        patch_size=16, embed_dim=embed_dim, backbone_depth=12, num_heads=6, drop_path_rate=0.2, 
        **kwargs
        )
    model = FusionNet(**model_kwargs)
    fromViT = False
    if pretrained_from == "ViT":
        fromViT=True
    if pretrained:
        model = load_checkpoint(model, pretrained_path, model.backbone_depth, model.context_depth,fromViT)
    return model

def funet_1img_52in1(pretrained=False, pretrained_path=None, num_classes=2, fusing=True, pretrained_from="ViT", context_depth=4, l_c=1, l_d=384,**kwargs):
    """ ViT-Tiny (Vit-Ti/2)
    """
    embed_dim = 384
    model_kwargs = dict(
        context_depth = context_depth,
        l_c = l_c,
        l_d = l_d,
        n_modalities = 2,
        channel_dims = [52,3],
        num_spatial_axes = [1,2], 
        out_dims = num_classes,
        num_classes = num_classes,
        fusing = fusing,
        patch_size=16, embed_dim=embed_dim, backbone_depth=12, num_heads=6, drop_path_rate=0.2, 
        **kwargs
        )
    model = FusionNet(**model_kwargs)
    fromViT = False
    if pretrained_from == "ViT":
        fromViT=True
    if pretrained:
        model = load_checkpoint(model, pretrained_path, model.backbone_depth, model.context_depth,fromViT)
    return model

def load_checkpoint(model, checkpoint_path, backbone_depth, context_depth, fromViT=True):
    #raise RuntimeError("Sqlist to Mdlist")
    import os
    from semilearn.nets.utils import resize_pos_embed_vit
    from torch.hub import load_state_dict_from_url
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = load_state_dict_from_url(checkpoint_path, map_location='cpu')

    layer_block_depth = backbone_depth//context_depth
    orig_state_dict = checkpoint['model']
    new_state_dict = {}
    for key, item in orig_state_dict.items():

        
        if key.startswith('module'):
            key = '.'.join(key.split('.')[1:])

        if fromViT and key.startswith('blocks'):
            keylist = key.split('.') # "blocks.0.ff"
            squeeze = int(keylist[1])
            unsqueeze = [str(squeeze//layer_block_depth), str(squeeze%layer_block_depth)]
            key = ".".join(keylist[0:1]+unsqueeze+keylist[2:]) # "blocks.0.0.ff"
        
        # TODO: better ways
        if key.startswith('fc') or key.startswith('classifier') or key.startswith('mlp') or key.startswith('head') or key.startswith('layers') or key.startswith('latents'):
            continue
            
        # check vit and interpolate
        # if isinstance(model, VisionTransformer) and 'patch_emb'

        if fromViT and key == 'pos_embed':
            posemb_new = model.pos_embed.data
            posemb = item
            item = resize_pos_embed_vit(posemb, posemb_new)

        new_state_dict[key] = item 
    
    match = model.load_state_dict(new_state_dict, strict=False)
    print(match)
    return model

class latent_cluster(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass