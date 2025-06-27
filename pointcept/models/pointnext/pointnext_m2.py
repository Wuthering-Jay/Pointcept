"""
Point Transformer V2 Mode 5 (recommend)

Disable Grouped Linear

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from copy import deepcopy
import torch
import torch.nn as nn
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
    
    
class LocalAggregation(nn.Module):
    """ 局部特征聚合层
    使用共享的MLP网络对点云特征进行聚合
    """
    def __init__(self, embed_channels):
        super().__init__()
        # 定义共享MLP网络（使用Linear层）
        self.embed_channels=embed_channels
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_channels, embed_channels),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
            nn.Linear(embed_channels, embed_channels),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, feat, coord, reference_index):
        """
        前向传播
        
        参数:
            feat: 输入特征 [n, c]
            coord: 点坐标 [n, 3]
            reference_index: 邻域索引 [n, k]
            
        返回:
            [n, c] 输出特征
        """
        grouped_features = pointops.grouping(reference_index, feat, coord, with_xyz=False)
        n_points = grouped_features.shape[0]
        grouped_features = grouped_features.reshape(-1, grouped_features.shape[-1])
        out = self.shared_mlp(grouped_features)
        out = out.reshape(n_points, -1, self.embed_channels)
        out = out.max(dim=1)[0]
        
        return out  # [n, c]


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
    """
    def __init__(
        self,
        embed_channels,
        drop_path_rate=0.0,
    ):
        super(Block, self).__init__()

        self.local_aggregation=LocalAggregation(embed_channels=embed_channels)
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
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
        feat = self.local_aggregation(feat, coord, reference_index) # [n, c]
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
        neighbours=16,
        drop_path_rate=0.0,
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
                drop_path_rate=drop_path_rates[i],
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
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        _, sorted_cluster_indices = torch.sort(cluster) # 格网化索引
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)]) # [v+1], 体素分段
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean") # [v, 3], 坐标平均池化
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max") # [v, c], 特征最大池化
        batch = batch[idx_ptr[:-1]] # [v]
        offset = batch2offset(batch) # [v] -> [b]
        return [coord, feat, offset], cluster


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
        backend="map",
    ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

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

    def forward(self, points, skip_points, cluster=None):
        """
        input: points: [pxo], [[n,3],[n,c],[b]], skip_points: [pxo], [[ns,3],[ns,c],[b]], cluster: [ns]
        output: points: [pxo], [[ns,3],[ns,c],[b]]
        """
        coord, feat, offset = points # [n, 3] [n, c] [b]
        skip_coord, skip_feat, skip_offset = skip_points # [ns, 3] [ns, c] [b]
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster] # [n, c] -> [ns, c], 投影上采样
        else:
            feat = pointops.interpolation(
                coord, skip_coord, self.proj(feat), offset, skip_offset
            ) # [n, c] -> [ns, c], 插值上采样
        if self.skip: # 跳跃连接，特征融合
            feat = feat + self.proj_skip(skip_feat) # [ns, c]
        return [skip_coord, feat, skip_offset] # [ns, 3] [ns, c] [b]


class Encoder(nn.Module):
    """
    Encoder for Point Transformer V2, 先进行格网池化, 再进行BlockSequence处理
    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        grid_size: 体素大小
        neighbours: 邻域大小
        drop_path_rate: BottleNeck的drop比例
    """
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        grid_size=None,
        neighbours=16,
        drop_path_rate=None,
    ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            neighbours=neighbours,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
        )

    def forward(self, points):
        """
        input: points: [pxo], [[n,3],[n,c],[b]]
        output: points: [pxo], [[ns,3],[ns,c],[b]], cluster: [n]
        """
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    """
    Decoder for Point Transformer V2, 先进行上采样, 再进行BlockSequence处理
    Args:
        in_channels: 输入维度
        skip_channels: 跳跃连接维度
        embed_channels: 输出维度
        depth: 解码器深度
        neighbours: 邻域大小
        drop_path_rate: BottleNeck的drop比例
        unpool_backend: 上采样方式，'map' or 'interp'
    """
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        depth,
        neighbours=16,
        drop_path_rate=None,
        unpool_backend="map",
    ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            neighbours=neighbours,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
        )

    def forward(self, points, skip_points, cluster):
        """
        input: points: [pxo], [[ns,3],[ns,c],[b]], skip_points: [pxo], [[n,3],[n,c],[b]], cluster: [n]
        output: points: [pxo], [[n,3],[n,c],[b]]
        """
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    """
    Patch Embedding for Point Transformer V2
    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        neighbours: 邻域大小
        drop_path_rate: BottleNeck的drop比例
    """
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        neighbours=16,
        drop_path_rate=0.0,
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
            neighbours=neighbours,
            drop_path_rate=drop_path_rate,
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


@MODELS.register_module("PNext-m2")
class PointNext(nn.Module):
    """
    Point Transformer V2
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
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.12, 0.24, 0.48),
        drop_path_rate=0,
        unpool_backend="map",
    ):
        super(PointNext, self).__init__()
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
        assert self.num_stages == len(grid_sizes)
        # 点云嵌入层
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
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
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                drop_path_rate=enc_dp_rates[
                    sum(enc_depths[:i]) : sum(enc_depths[: i + 1])
                ],
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                neighbours=dec_neighbours[i],
                drop_path_rate=dec_dp_rates[
                    sum(dec_depths[:i]) : sum(dec_depths[: i + 1])
                ],
                unpool_backend=unpool_backend,
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
        skips = [[points]] # 便于添加cluster
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling, 记录池化时的格网索引
            skips.append([points])  # record points info of current stage, 记录池化后的当前点云信息
        # 此时skips共五层，最后一层不带有cluster信息
        # 取出最后一层的点云信息
        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster) # 上采样
        coord, feat, offset = points
        seg_logits = self.seg_head(feat) # [n, num_classes]
        return seg_logits
