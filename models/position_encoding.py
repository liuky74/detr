# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    这个位置嵌入模块不包含任何torch下的模块,全部的操作都在forward中完成
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi #两倍Π
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 统计height维度上的累加,功能上即像素索引idx
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 统计width维度上的累加
        if self.normalize: # 将像素索引均值化
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        # num_pos_feats为每个像素的特征维度,这里建立了特征维度的索引(注意与像素维度的索引不同),区分特征维度是为了给每个像素赋予不同的坐标信息.
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        ''' 2i/d, i=dim_t//2,这里对2整除是因为下面每2个特征维度为一组,一组中一个进行sin变换,一个进行cos变换,
            整除可以保证每一组每的两个特征维度的索引是相同的,d是特征的维度数量,/d是为了使不同维度拥有不同的跨度,
            论文中对于位置的编码公式为 PE(pos,2i)=sin(position/(10000^(2i/d))), 因为每个维度的i是不同的,
            因此随着维度增大分母也会增大,position会映射到不同跨度的sin函数上,从而达到不同的位置编码.
            (也就是用d个函数来表达位置信息,从而扩大位置编码的表示范围.)
            '''
        dim_t = (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** dim_t  # 10000 ^ (2*dim_t/2/128)
        pos_x = x_embed[:, :, :, None] / dim_t  # 将像素索引映射到dim_t域内,表示了每个特征维度的x轴坐标索引
        pos_y = y_embed[:, :, :, None] / dim_t # 将像素索引映射到dim_t域内,表示了每个特征维度的y轴坐标索引
        # cos和sin操作,其实哪些维度用sin哪些用cos并不重要,这只是让编码更[丰富]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True) # 坐标嵌入
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
