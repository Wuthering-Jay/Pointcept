"""
PointNext mode 2 for Semantic Segmentation using Grouped Vector Attention (GVA)
"""

import torch
import torch.nn as nn
import einops
import pointops

from pointcept.models.builder import MODELS


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    对形状为[n, c], [n, l, c]的点云数据进行批量归一化
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError
        

class PointTransformerLayer(nn.Module):
    """
    点云变换模块
    Args:
        in_planes: 输入特征维度
        out_planes: 输出特征维度
        share_planes: 共享特征维度
        nsample: 每个点采样的点数
    """
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            PointBatchNorm(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            PointBatchNorm(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            PointBatchNorm(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        """
        input: pxo: (n, 3), (n, c), (b)
        output: x: (n, out_planes)
        """
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x) # (n, mid), (n, mid), (n, out)
        x_k, idx = pointops.knn_query_and_group(
            x_k, p, o, new_xyz=p, new_offset=o, nsample=self.nsample, with_xyz=True
        ) # (n, nsample, 3+mid), (n, nsample)
        x_v, _ = pointops.knn_query_and_group(
            x_v,
            p,
            o,
            new_xyz=p,
            new_offset=o,
            idx=idx,
            nsample=self.nsample,
            with_xyz=False,
        ) # (n, nsample, out)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:] # (n, nsample, 3), (n, nsample, mid)
        p_r = self.linear_p(p_r) # (n, nsample, out)
        r_qk = (
            x_k # (n, nsample, mid)
            - x_q.unsqueeze(1) # (n, 1, mid)
            + einops.reduce(
                p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes # (n, nsample, mid)
            ) # (n, nsample, mid)
        )
        w = self.linear_w(r_qk)  # (n, nsample, c)
        w = self.softmax(w) # (n, nsample, c)
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes), # (n, nsample, out) -> (n, nsample, share, out/share)
            w,
        ) # (n, share, out/share)
        x = einops.rearrange(x, "n s i -> n (s i)") # (n, out)
        return x
    

class TransitionDown(nn.Module):
    """
    点云降采样模块
    Args:
        in_planes: 输入特征维度
        out_planes: 输出特征维度
        stride: 降采样率
        nsample: 每个点采样的点数
    """
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        """
        input: pxo: (n, 3), (n, c), (b)
        output: pxo: (m, 3), (m, out), (b)
        """
        p, x, o = pxo  # (n, 3), (n, c), (b)
        # stride不为1时，降采样
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o) # 降采样的 offset
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x, _ = pointops.knn_query_and_group(
                x,
                p,
                offset=o,
                new_xyz=n_p,
                new_offset=n_o,
                nsample=self.nsample,
                with_xyz=True,
            ) # (m, nsample, 3+c)
            x = self.relu(
                self.bn(self.linear(x).transpose(1, 2).contiguous())
            )  # (m, out, nsample)
            x = self.pool(x).squeeze(-1)  # (m, out)
            p, o = n_p, n_o # (m, 3), (b)
        # stride为1时，直接线性变换
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, out)
        return [p, x, o]


class TransitionUp(nn.Module):
    """
    点云上采样模块
    Args:
        in_planes: 输入特征维度
        out_planes: 输出特征维度
    """
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )

    def forward(self, pxo1, pxo2=None):
        """
        input: pxo1: (n, 3), (n, c), (b)
               pxo2: (m, 3), (m, c), (b)
        output: x: (n, out_planes)
        """
        # pxo2为None时，直接线性变换
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]): # 逐offset
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0] # start, end, count
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :] # (cnt, c)
                x_b = torch.cat(
                    (x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1 # 点特征 + batch全局特征
                )
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        # pxo2不为None时，进行插值
        else:
            p1, x1, o1 = pxo1 # (n, 3), (n, c), (b)
            p2, x2, o2 = pxo2 # (m, 3), (m, c), (b)
            x = self.linear1(x1) + pointops.interpolation(
                p2, p1, self.linear2(x2), o2, o1 # 默认为3个临近点插值
            ) # (n, c)
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck block
    Args:
        in_planes: 输入特征维度
        planes: 输出特征维度
        nsample: 每个点采样的点数
    """
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        """
        input: pxo: (n, 3), (n, c), (b)
        output: pxo: (n, 3), (n, c), (b)
        """
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x # (n, c)
        x = self.relu(self.bn1(self.linear1(x))) # (n, planes)
        x = self.relu(self.bn2(self.transformer([p, x, o]))) # (n, planes)
        x = self.bn3(self.linear3(x)) # (n, planes*expansion)
        x += identity # (n, c)
        x = self.relu(x) # (n, c)
        return [p, x, o] # (n, 3), (n, c), (b)


@MODELS.register_module("PT-v1m1")
class PointTransformer(nn.Module):
    """
    Point Transformer for Semantic Segmentation (Dynamic Version)

    The number of layers is now determined by the length of the 'blocks',
    'planes', 'stride', and 'nsample' lists.

    Args:
        block: Block type, defaults to Bottleneck.
        blocks (list[int]): Number of blocks at each layer. The length of this
                           list determines the depth of the network.
        planes (list[int]): Number of output feature planes at each layer.
        stride (list[int]): Downsampling stride at each layer.
        nsample (list[int]): Number of points to sample in each neighborhood.
        in_channels (int): Input feature dimension.
        num_classes (int): Number of output classes.
    """
    def __init__(self,
                 block=Bottleneck,
                 enc_blocks=[1, 1, 1, 1, 1],
                 dec_blocks=[1, 1, 1, 1, 1],
                 planes=[32, 64, 128, 256, 512],
                 share_planes=8,
                 stride=[1, 4, 4, 4, 4],
                 nsample=[8, 16, 16, 16, 16],
                 in_channels=6,
                 num_classes=13):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = len(enc_blocks)

        # Ensure all configuration lists have the same length
        assert self.num_layers == len(dec_blocks) == len(planes) == len(stride) == len(nsample), \
            "The length of blocks, planes, stride, and nsample must be the same."

        # Encoder
        self.enc_modules = nn.ModuleList()
        self.in_planes = self.in_channels
        for i in range(self.num_layers):
            self.enc_modules.append(
                self._make_enc(
                    block,
                    planes[i],
                    enc_blocks[i],
                    stride=stride[i],
                    nsample=nsample[i],
                    share_planes=share_planes
                )
            )

        # Decoder
        # The decoder processes layers in reverse order (from deep to shallow)
        self.dec_modules = nn.ModuleList()
        for i in range(self.num_layers - 1, -1, -1):
            is_head = (i == self.num_layers - 1)
            self.dec_modules.append(
                self._make_dec(
                    block,
                    planes[i],
                    dec_blocks[i], # Typically 1 block in decoder stages
                    nsample=nsample[i],
                    is_head=is_head,
                    share_planes=share_planes
                )
            )
        
        # Final classification layer
        # The input dimension is the output of the final decoder stage
        self.cls = nn.Sequential(
            nn.Linear(planes[0] * block.expansion, planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[0], num_classes),
        )

    def _make_enc(self, block, planes, blocks, stride=1, nsample=16, share_planes=8):
        """
        Creates an encoder layer.
        """
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, nsample=nsample, share_planes=share_planes)
            )
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, nsample=16, is_head=False, share_planes=8):
        """
        Creates a decoder layer.
        """
        layers = [
            TransitionUp(self.in_planes, None if is_head else planes * block.expansion)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, nsample=nsample, share_planes=share_planes)
            )
        return nn.Sequential(*layers)

    def forward(self, data_dict):

        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()

        enc_outputs = []
        p, x, o = p0, x0, o0

        for i in range(self.num_layers):
            p, x, o = self.enc_modules[i]([p, x, o])
            enc_outputs.append((p, x, o))

        p, x, o = enc_outputs[-1]
        x_prev = self.dec_modules[0][1:]([p, self.dec_modules[0][0]([p, x, o]), o])[1]
        p_prev, o_prev = p, o

        for i in range(1, self.num_layers):
            dec_module = self.dec_modules[i]
            p_skip, x_skip, o_skip = enc_outputs[-(i + 1)]
            x_up = dec_module[0]([p_skip, x_skip, o_skip], [p_prev, x_prev, o_prev])
            x_prev = dec_module[1:]([p_skip, x_up, o_skip])[1]
            p_prev, o_prev = p_skip, o_skip

        out = self.cls(x_prev)
        return out

"""
Point Transformer V2 Mode 5 (recommend)

Disable Grouped Linear

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

import einops
from timm.layers import DropPath
import pointops

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    对形状为[n, c], [n, l, c]的点云数据进行批量归一化
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError
        
        
class PointNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k_neighbors=16):
        """
        使用 Linear 层和 PointBatchNorm 的 PointNet 层实现
        
        参数:
            in_channels: 输入特征维度 c1
            out_channels: 输出特征维度 c2
            k_neighbors: KNN近邻数
        """
        super().__init__()
        self.k_neighbors = k_neighbors
        
        # 计算中间层维度
        mid_channels = max(in_channels, out_channels // 2)
        
        # 输入特征维度调整（如果包含xyz坐标）
        mlp_in_channels = in_channels
        
        # 定义共享MLP网络（使用Linear层）
        self.shared_mlp = nn.Sequential(
            nn.Linear(mlp_in_channels, mid_channels),
            PointBatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_channels),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, points):
        """
        前向传播
        
        参数:
            x: 输入特征 [n, c1]
            coord: 点坐标 [n, 3]
            offset: 点云偏移量
            
        返回:
            [n, c2] 输出特征
        """
        coord, feat, offset = points
        reference_index, _ = pointops.knn_query(self.k_neighbors, coord, offset)
        grouped_features = pointops.grouping(reference_index, feat, coord, with_xyz=False)
        n_points = grouped_features.shape[0]
        grouped_features = grouped_features.reshape(-1, grouped_features.shape[-1])
        out = self.shared_mlp(grouped_features)
        out = out.reshape(n_points, self.k_neighbors, -1)
        out = out.max(dim=1)[0]
        
        return out


class GroupedVectorAttention(nn.Module):
    """
    分组向量注意力机制
    Args:
        embed_channesl: 输入输出维度
        groups: 分组数量
        attn_drop_rate: drop比例
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
    """
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
    ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        """
        input: feat: [n, c], coord: [n, 3], reference_index: [n, k]
        output: feat: [n, c]
        """
        query, key, value = (
            self.linear_q(feat), # [n, c]
            self.linear_k(feat), # [n, c]
            self.linear_v(feat), # [n, c]
        )
        key = pointops.grouping(reference_index, key, coord, with_xyz=True) # [n, k, 3+c]
        value = pointops.grouping(reference_index, value, coord, with_xyz=False) # [n, k, c]
        pos, key = key[:, :, 0:3], key[:, :, 3:] # [n, k, 3], [n, k, c]
        relation_qk = key - query.unsqueeze(1) # [n, k ,c], 邻域内与中心点的相对位置, 用于相对位置编码
        if self.pe_multiplier: # 乘性因子
            pem = self.linear_p_multiplier(pos) # [n, k, c]
            relation_qk = relation_qk * pem # [n, k, c]
        if self.pe_bias: # 偏置因子
            peb = self.linear_p_bias(pos) # [n, k, c]
            relation_qk = relation_qk + peb # [n, k, c]
            value = value + peb # [n, k, c]

        weight = self.weight_encoding(relation_qk) # [n, k, g]
        weight = self.attn_drop(self.softmax(weight)) # [n, k, g]

        mask = torch.sign(reference_index + 1) # [n, k], 无效邻域点标记为0
        weight = torch.einsum("n s g, n s -> n s g", weight, mask) # [n, k, g]
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups) # [n, k, g, i]
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight) # [n, g, i]
        feat = einops.rearrange(feat, "n g i -> n (g i)") # [n, c]
        return feat # [n, c]


class Block(nn.Module):
    """
    网络模块单位，结合GVA和BottleNeck对pxo进行处理，不改变数据维度
    Args:
        embed_channesl: 输入输出维度
        groups: 分组数量
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制，以时间换空间
    """
    def __init__(
        self,
        embed_channels,
        groups,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, points, reference_index):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], reference_index: [n, k]
        output: [pxo], [[n,3],[n,c],[b]], 不改变维度
        """
        coord, feat, offset = points # [n,3], [n,c], [b]
        identity = feat # [n, c]
        feat = self.act(self.norm1(self.fc1(feat))) # [n, c]
        feat = (
            self.attn(feat, coord, reference_index)
            if not self.enable_checkpoint # checkpoint机制，时间换空间，梯度等部分参数不保留，在反向传播时重新计算
            else checkpoint(self.attn, feat, coord, reference_index)
        ) # [n, c]
        feat = self.act(self.norm2(feat)) # [n, c]
        feat = self.norm3(self.fc3(feat)) # [n, c]
        feat = identity + self.drop_path(feat) # [n, c], bottleneck设计
        feat = self.act(feat) # [n, c]
        return [coord, feat, offset] # [[n,3],[n,c],[b]]


class BlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(BlockSequence, self).__init__()

        # 确保 drop_path_rates 为 list
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        # 多个 Block 堆叠
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points):
        """
        input: points: [pxo], [[n,3],[n,c],[b]]
        output: [pxo], [[n,3],[n,c],[b]]
        """
        coord, feat, offset = points 
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset) # [n, k]
        for block in self.blocks:
            points = block(points, reference_index)
        return points


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    格网池化，基于体素划分进行池化下采样，体素内坐标平均池化，特征最大池化，得到新的pxo，同时输出体素索引
    Args:
        in_channels: 输入维度
        out_channels: 输出维度
        grid_size: 体素大小
        bias: fc层偏置
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], start: [b, 3]
        output: points: [pxo], [[v,3],[v,c],[b]], cluster: [n]
        """
        coord, feat, offset = points # [n, 3] [n, c] [b]
        batch = offset2batch(offset) # [b] -> [n]
        feat = self.act(self.norm(self.fc(feat))) # [n, c]
        start = (
            segment_csr(
                coord,
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                reduce="min",
            ) # [b, 3], 求每个batch的最小点
            if start is None
            else start
        )
        cluster = voxel_grid(
            pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
        ) # [n], 计算每个点在网格中的索引
        _, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        _, sorted_cluster_indices = torch.sort(cluster) # 格网化索引
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)]) # [v+1], 体素分段
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean") # [v, 3], 坐标平均池化
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max") # [v, c], 特征最大池化
        batch = batch[idx_ptr[:-1]] # [v]
        offset = batch2offset(batch) # [v] -> [b]
        return [coord, feat, offset]


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    带有跳跃连接的上采样
    Args:
        in_channels: 输入维度
        out_channels: 输出维度
        skip_channels: 跳跃连接维度
        bias: fc层偏置
        skip: 是否使用跳跃连接
        backend: 上采样方式，'map' or 'interp'
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        bias=True,
        skip=True,
    ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], skip_points: [pxo], [[ns,3],[ns,c],[b]]
        output: points: [pxo], [[ns,3],[ns,c],[b]]
        """
        coord, feat, offset = points # [n, 3] [n, c] [b]
        skip_coord, skip_feat, skip_offset = skip_points # [ns, 3] [ns, c] [b]

        feat = pointops.interpolation(
            coord, skip_coord, self.proj(feat), offset, skip_offset
        ) # [n, c] -> [ns, c], 插值上采样
        if self.skip: # 跳跃连接，特征融合
            feat = feat + self.proj_skip(skip_feat) # [ns, c]
        return [skip_coord, feat, skip_offset] # [ns, 3] [ns, c] [b]
    
# 添加TransitionDown替换GridPool
class TransitionDown(nn.Module):
    """
    点云降采样模块，使用最远点采样
    Args:
        in_planes: 输入特征维度
        out_planes: 输出特征维度
        stride: 降采样率
        nsample: 每个点采样的点数
    """
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, points):
        """
        input: points: [pxo], [[n,3],[n,c],[b]]
        output: points: [pxo], [[m,3],[m,c],[b]]
        """
        p, x, o = points  # (n, 3), (n, c), (b)
        # stride不为1时，降采样
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o) # 降采样的 offset
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x, _ = pointops.knn_query_and_group(
                x,
                p,
                offset=o,
                new_xyz=n_p,
                new_offset=n_o,
                nsample=self.nsample,
                with_xyz=True,
            ) # (m, nsample, 3+c)
            x = self.relu(
                self.bn(self.linear(x).transpose(1, 2).contiguous())
            )  # (m, out, nsample)
            x = self.pool(x).squeeze(-1)  # (m, out)
            p, o = n_p, n_o # (m, 3), (b)
        # stride为1时，直接线性变换
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, out)
        return [p, x, o]

# 添加TransitionUp替换UnpoolWithSkip
class TransitionUp(nn.Module):
    """
    点云上采样模块，使用特征插值
    Args:
        in_planes: 输入特征维度
        out_planes: 输出特征维度
    """
    def __init__(self, in_planes, out_planes):
        super().__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(out_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, pxo1, pxo2=None):
        """
        input: pxo1: [pxo], [[n,3],[n,c],[b]]  (低分辨率点云)
               pxo2: [pxo], [[m,3],[m,c],[b]]  (高分辨率点云，跳跃连接)
        output: feat: [m, out_planes]  (上采样后的特征)
        """
        # pxo2为None时，只处理单个点云
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]): # 逐offset
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0] # start, end, count
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :] # (cnt, c)
                x_b = torch.cat(
                    (x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1 # 点特征 + batch全局特征
                )
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
            return x
        # pxo2不为None时，进行跨点云特征插值
        else:
            p1, x1, o1 = pxo1 # (n, 3), (n, c), (b) - 低分辨率点云
            p2, x2, o2 = pxo2 # (m, 3), (m, c), (b) - 高分辨率点云
            x = self.linear1(x2) + pointops.interpolation(
                p1, p2, self.linear2(x1), o1, o2 # 从低分辨率点云插值到高分辨率点云
            ) # (m, c)
            return [p2, x, o2]  # 返回高分辨率点云的坐标和新特征


class Encoder(nn.Module):
    """
    Encoder for Point Transformer V2, 先进行格网池化, 再进行BlockSequence处理
    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        groups: 分组数量
        grid_size: 体素大小
        neighbours: 邻域大小
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制，以时间换空间
    """
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        stride=4,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
    ):
        super(Encoder, self).__init__()

        self.down = TransitionDown(
            in_planes=in_channels,
            out_planes=embed_channels,
            stride=stride,
            nsample=neighbours,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        """
        input: points: [pxo], [[n,3],[n,c],[b]]
        output: points: [pxo], [[ns,3],[ns,c],[b]]
        """
        points = self.down(points)
        return self.blocks(points)


class Decoder(nn.Module):
    """
    Decoder for Point Transformer V2, 先进行上采样, 再进行BlockSequence处理
    Args:
        in_channels: 输入维度
        skip_channels: 跳跃连接维度
        embed_channels: 输出维度
        groups: 分组数量
        depth: 解码器深度
        neighbours: 邻域大小
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制，以时间换空间
        unpool_backend: 上采样方式，'map' or 'interp'
    """
    def __init__(
        self,
        in_channels,
        embed_channels,
        groups,
        depth,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
    ):
        super(Decoder, self).__init__()

        self.up = TransitionUp(
            in_planes=in_channels,
            out_planes=embed_channels,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, skip_points):
        """
        input: points: [pxo], [[ns,3],[ns,c],[b]], skip_points: [pxo], [[n,3],[n,c],[b]]
        output: points: [pxo], [[n,3],[n,c],[b]]
        """
        points = self.up(points, skip_points)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    """
    Patch Embedding for Point Transformer V2
    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        groups: 分组数量
        neighbours: 邻域大小
        qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制，以时间换空间
    """
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = embed_channels // 2
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, self.mid_channels, bias=False),
            PointBatchNorm(self.mid_channels),
            nn.ReLU(inplace=True),
        )
        self.pointnet = PointNetLayer(in_channels, embed_channels-self.mid_channels, neighbours)
        
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        """
        input: points: [pxo], [[n,3],[n,c],[b]]
        output: points: [pxo], [[n,3],[n,c],[b]]
        """
        coord, feat, offset = points
        feat1 = self.proj(feat)
        feat2 = self.pointnet(points)
        feat = torch.cat([feat1,feat2],dim=1)
        return self.blocks([coord, feat, offset])


@MODELS.register_module("PT-v1m2")
class PointTransformerV1(nn.Module):
    """
    Point Transformer V1
    Args:
        in_channels: 输入维度
        num_classes: 输出维度
        patch_embed_depth: Patch Embedding深度
        patch_embed_channels: Patch Embedding输出维度
        patch_embed_groups: Patch Embedding分组数量
        patch_embed_neighbours: Patch Embedding邻域大小
        enc_depths: 编码器深度
        enc_channels: 编码器输出维度
        enc_groups: 编码器分组数量
        enc_neighbours: 编码器邻域大小
        dec_depths: 解码器深度
        dec_channels: 解码器输出维度
        dec_groups: 解码器分组数量
        dec_neighbours: 解码器邻域大小
        grid_sizes: 体素大小
        attn_qkv_bias: 无用
        pe_multiplier: 位置编码乘性因子
        pe_bias: 位置编码偏置因子
        attn_drop_rate: drop比例
        drop_path_rate: BottleNeck的drop比例
        enable_checkpoint: checkpoint机制，以时间换空间
        unpool_backend: 上采样方式，'map' or 'interp'
    """
    def __init__(
        self,
        in_channels,
        num_classes,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        strides=(4, 4, 4, 4),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        enable_checkpoint=False,
    ):
        super(PointTransformerV1, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(strides)
        # 点云嵌入层
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
        )
        # bottleneck的drop率逐渐提高
        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        dec_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))
        ]
        # 前一层的输出维度作为下一层的输入维度
        enc_channels = [patch_embed_channels] + list(enc_channels) # [48, 96, 192, 384, 512]
        dec_channels = list(dec_channels) + [enc_channels[-1]] # [48, 96, 192, 384, 512]
        # 编码器与解码器
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                stride=strides[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[
                    sum(enc_depths[:i]) : sum(enc_depths[: i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[
                    sum(dec_depths[:i]) : sum(dec_depths[: i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)
        # 分割头
        self.seg_head = (
            nn.Sequential(
                nn.Linear(dec_channels[0], dec_channels[0]),
                PointBatchNorm(dec_channels[0]),
                nn.ReLU(inplace=True),
                nn.Linear(dec_channels[0], num_classes),
            )
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, data_dict):
        """
        input: data_dict: {"coord": [n, 3], "feat": [n, c], "offset": [b]}
        output: seg_logits: [n, num_classes]
        """
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()

        # a batch of point cloud is a list of coord, feat and offset
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [points]
        
        # 编码阶段
        for i in range(self.num_stages):
            points = self.enc_stages[i](points)  # 不再接收cluster信息
            skips.append(points)  # 记录每个编码阶段的点云信息
        
        # 解码阶段
        points = skips.pop(-1)  # 获取最深层点云信息
        for i in reversed(range(self.num_stages)):
            skip_points = skips.pop(-1)  # 获取跳跃连接点云信息
            points = self.dec_stages[i](points, skip_points)  # 上采样并进行特征融合
        
        coord, feat, offset = points
        seg_logits = self.seg_head(feat)  # [n, num_classes]
        return seg_logits
