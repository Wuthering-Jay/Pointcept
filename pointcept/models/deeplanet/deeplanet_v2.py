"""
DeepLANet V2 Backbone

Backbone only (符合 DefaultSegmentorV2 规范):
  - 输入: data_dict 或 Point 对象
  - 输出: Point 对象 (feat 为解码器最终输出特征)

核心创新:
  1. Stage-Level Position Embedding: 每个 Stage 只计算一次位置编码
  2. Vector Feature Representation (VFR): 轻量级局部特征聚合，O(N) 复杂度
  3. ResLFE Block: 残差局部特征提取块，替代复杂的 LFA
     - Front-Linear: 先降维再 grouping，节省显存
     - PE 通过加法融合，不翻倍通道
     - 维度保持不变，更高效

V2 vs V1:
  - V1: LocalFeatureAggregation (重，两轮 LSE+Pool，输出 2*d_out)
  - V2: VFRModule (轻，仅边缘特征+max pooling，输出 d)
"""

from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

from timm.layers import DropPath
import pointops

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    对形状为[n, c], [n, l, c]的点云数据进行批量归一化
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Prevent AMP NaN by enforcing FP32 computation for BatchNorm
        input_dtype = input.dtype
        input_fp32 = input.float()
        if input_fp32.dim() == 3:
            out = (
                self.norm(input_fp32.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input_fp32.dim() == 2:
            out = self.norm(input_fp32)
        else:
            raise NotImplementedError
        return out.to(input_dtype)


def compute_stage_positional_encoding(coord, reference_index, dist):
    """
    阶段级位置编码计算（Stage-Level Position Embedding）

    核心创新: 在每个 Stage 开始时计算一次位置编码，所有 Block 共享，
    避免每层重复计算，极大节约计算量。

    位置编码包含:
    - 中心点坐标 (3)
    - 邻域点坐标 (3)
    - 相对位置 (3)
    - 距离 (1)
    共 10 维

    Args:
        coord: [n, 3] 点云坐标
        reference_index: [n, k] 邻域索引
        dist: [n, k] 邻域距离

    Returns:
        pos_encoding: [n, k, 10] 位置编码特征
    """
    n, k = reference_index.shape

    # 获取邻域坐标: [n, k, 3]
    neighbor_coord = pointops.grouping(reference_index, coord, coord, with_xyz=False)

    # 中心点坐标扩展: [n, 3] -> [n, k, 3]
    center_coord = coord.unsqueeze(1).expand(-1, k, -1)

    # 相对位置: [n, k, 3]
    relative_pos = center_coord - neighbor_coord

    # 距离: [n, k] -> [n, k, 1]
    dist_feat = dist.unsqueeze(-1)

    # 拼接位置编码: [n, k, 10]
    pos_encoding = torch.cat([
        center_coord,      # [n, k, 3]
        neighbor_coord,    # [n, k, 3]
        relative_pos,      # [n, k, 3]
        dist_feat,         # [n, k, 1]
    ], dim=-1)

    return pos_encoding


class PositionalEncodingEncoder(nn.Module):
    """
    位置编码编码器
    将 Stage-Level 计算的位置编码 [n, k, 10] 编码为 [n, C] 用于加法融合

    Args:
        embed_channels: 输出特征维度
    """

    def __init__(self, embed_channels):
        super(PositionalEncodingEncoder, self).__init__()
        # 10维 -> embed_channels，然后 max pooling
        self.mlp = nn.Sequential(
            nn.Linear(10, embed_channels),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pos_encoding):
        """
        Args:
            pos_encoding: [n, k, 10] Stage级别的位置编码

        Returns:
            [n, C] 编码后的位置特征，用于与点特征相加
        """
        import torch
        n, k, _ = pos_encoding.shape

        # 强制剥离amp并升为fp32清洗极值与Inf，彻底避免在深层网络(Stage 4)遇到极限距离时Linear层溢出为NaN
        origin_dtype = pos_encoding.dtype
        pos_encoding_fp32 = pos_encoding.float()
        pos_encoding_fp32 = torch.nan_to_num(pos_encoding_fp32, nan=0.0, posinf=65000.0, neginf=-65000.0)

        # MLP 编码在彻底关闭 AMP 的浮点上下文下运算: [n, k, 10] -> [n, k, C]
        with torch.amp.autocast('cuda', enabled=False):
            pe_feat = self.mlp(pos_encoding_fp32.reshape(-1, 10)).reshape(n, k, -1)

        # Max pooling 聚合邻域信息: [n, k, C] -> [n, C]
        pe_feat = pe_feat.max(dim=1)[0]

        return pe_feat.to(origin_dtype)


class VFRModule(nn.Module):
    """
    Vector Feature Representation Module (向量特征表示模块)

    核心创新: 仅做局部差值和最大池化，O(N) 复杂度，极轻量
    """

    def __init__(self):
        super(VFRModule, self).__init__()
        # 注意：这里没有 Linear，因为 Linear 被前置 (Front-Linear) 以降低维度

    def forward(self, feat, coord, reference_index):
        """
        Args:
            feat: [n, C] 点云特征
            coord: [n, 3] 点云坐标
            reference_index: [n, k] 邻域索引

        Returns:
            [n, C] 聚合后的特征
        """
        # 获取邻居特征: [n, k, C]
        # 使用 amp 禁止的话可以避免极端浮点溢出，由于这里的差值较小影响不大
        grouped_feat = pointops.grouping(reference_index, feat, coord, with_xyz=False)

        # 核心: 计算边缘特征 f_j - f_i (相对特征)
        rel_feat = grouped_feat - feat.unsqueeze(1)  # [n, k, C]

        # 最大池化聚合: [n, k, C] -> [n, C]
        out_feat = rel_feat.max(dim=1)[0]
        
        # 加上原特征的一个残差，防止0值累积导致后续为NaN
        return out_feat + feat


class ResLFEBlock(nn.Module):
    """
    Residual Local Feature Extraction Block (残差局部特征提取块)

    核心创新: 彻底替代 V1 中复杂的 LocalFeatureAggregation
    1. Front-Linear: 先降维/投影，再去 Grouping (节省显存)
    2. PE 融合: 通过加法直接融合，不翻倍通道
    3. VFR: 极轻量级的局部聚合（仅边缘特征+max pooling）
    4. Modern Structure: 类似 Transformer FFN，但维度保持不变（1x BottleNeck）
    5. 深层网络优化: 零初始化 + LayerScale 提升训练稳定性

    Args:
        embed_channels: 输入输出维度
        drop_path_rate: DropPath 比例
        enable_checkpoint: checkpoint 机制
        enable_layer_scale: 是否启用 LayerScale (深层网络建议开启)
        layer_scale_init_value: LayerScale 初始值 (默认 1e-5，120层建议 1e-6)
    """

    def __init__(
        self,
        embed_channels,
        drop_path_rate=0.0,
        enable_checkpoint=False,
        enable_layer_scale=False,
        layer_scale_init_value=1e-5,
    ):
        super(ResLFEBlock, self).__init__()
        self.enable_checkpoint = enable_checkpoint
        self.enable_layer_scale = enable_layer_scale

        # 1. Front-Linear: 先降维/投影，再去 Grouping (关键省显存操作)
        self.norm1 = PointBatchNorm(embed_channels)
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)

        # 2. VFR: 极轻量级的局部聚合
        self.vfr = VFRModule()

        # 3. Modern Structure: 类似 Transformer FFN，但不放大维度 (1x BottleNeck)
        self.norm2 = PointBatchNorm(embed_channels)
        self.fc2 = nn.Linear(embed_channels, embed_channels, bias=False)

        # 【深层网络优化 1】零初始化: 强制残差分支初始化输出为 0
        # 让网络初始时接近恒等映射，训练更稳定
        nn.init.constant_(self.norm2.norm.weight, 0.0)

        self.act = nn.ReLU(inplace=True)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        # 【深层网络优化 2】LayerScale: 可学习的极小缩放因子
        # 在残差相加前乘以 gamma，压制深层残差幅度
        if self.enable_layer_scale:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(embed_channels)
            )

    def forward(self, points, pe, reference_index):
        """
        Args:
            points: [pxo], [[n,3],[n,c],[b]]
            pe: [n, C] 预编码的位置特征（Stage级别）
            reference_index: [n, k] 邻域索引

        Returns:
            [pxo], [[n,3],[n,c],[b]], 维度保持不变
        """
        coord, feat, offset = points
        identity = feat

        # Pre-Norm & Front-Linear
        feat = self.act(self.norm1(self.fc1(feat)))

        # 融合阶段级 PE (N x C 直接相加，不翻倍通道)
        feat = feat + pe.type_as(feat)

        # 局部信息提取
        feat = (
            self.vfr(feat, coord, reference_index)
            if not self.enable_checkpoint
            else checkpoint(self.vfr, feat, coord, reference_index, use_reentrant=False)
        )

        # Post-Norm & FFN (1x 维度保持不变)
        feat = self.norm2(self.fc2(feat))

        # 残差连接: 应用 LayerScale (如果启用)
        if self.enable_layer_scale:
            feat = identity + self.drop_path(feat * self.gamma.type_as(feat))
        else:
            feat = identity + self.drop_path(feat)
        feat = self.act(feat)

        return [coord, feat, offset]


class BlockSequence(nn.Module):
    """
    Block序列，多个 ResLFEBlock 堆叠
    基于 Stage-Level Position Embedding:
      1. 在 Stage 开始时计算一次 [n, k, 10] 位置编码
      2. 通过 PE encoder 编码为 [n, C]
      3. 所有 Block 共享，极大减少计算量

    深层网络优化 (零初始化 + LayerScale):
      - 所有残差块初始输出为 0 (零初始化)
      - LayerScale 可学习地缩放残差幅度 (可选)

    Args:
        depth: Block数量
        embed_channels: 特征维度
        neighbours: 邻域点数量
        drop_path_rate: DropPath比例
        enable_checkpoint: checkpoint机制
        enable_layer_scale: 是否启用 LayerScale (深层网络建议开启)
        layer_scale_init_value: LayerScale 初始值 (默认 1e-5，120层建议 1e-6)
    """

    def __init__(
        self,
        depth,
        embed_channels,
        neighbours=16,
        drop_path_rate=0.0,
        enable_checkpoint=False,
        enable_layer_scale=False,
        layer_scale_init_value=1e-5,
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

        # PE encoder: [n, k, 10] -> [n, C]
        self.pe_encoder = PositionalEncodingEncoder(embed_channels)

        # 多个 ResLFEBlock 堆叠
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = ResLFEBlock(
                embed_channels=embed_channels,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
                enable_layer_scale=enable_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
            self.blocks.append(block)

    def forward(self, points):
        """
        Stage-Level Position Embedding 优化:
        1. 计算一次 KNN 和原始位置编码 [n, k, 10]
        2. 编码为 [n, C] 的位置特征
        3. 所有 Block 共享

        Args:
            points: [pxo], [[n,3],[n,c],[b]]

        Returns:
            [pxo], [[n,3],[n,c],[b]]
        """
        coord, feat, offset = points

        # Stage级别: 计算一次 KNN 查询和位置编码
        with torch.no_grad():
            reference_index, dist = pointops.knn_query(self.neighbours, coord, offset)
            # 计算原始位置编码: [n, k, 10]
            pos_encoding = compute_stage_positional_encoding(coord, reference_index, dist)

        # 编码为点级别的位置特征: [n, k, 10] -> [n, C]
        pe = self.pe_encoder(pos_encoding)

        # 所有 Block 共享同一个位置特征和邻域索引
        for block in self.blocks:
            points = block(points, pe, reference_index)

        return points


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    格网池化，基于体素划分进行池化下采样

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
        Args:
            points: [pxo], [[n,3],[n,c],[b]]
            start: [b, 3]

        Returns:
            points: [pxo], [[v,3],[v,c],[b]]
            cluster: [n]
        """
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))

        import torch
        with torch.no_grad():
            start = (
                segment_csr(
                    coord,
                    torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                    reduce="min",
                )
                if start is None
                else start
            )
            cluster = voxel_grid(
                pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
            )
            unique, cluster, counts = torch.unique(
                cluster, sorted=True, return_inverse=True, return_counts=True
            )
            _, sorted_cluster_indices = torch.sort(cluster)
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        # clamp feature max range under float16 to avoid overflowing to inf
        feat_max = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        # to prevent half precision inf overflow in extreme cases
        if feat_max.dtype == torch.float16:
            feat_max = torch.clamp(feat_max, min=-65500.0, max=65500.0)
            
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat_max, offset], cluster.detach()


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
        Args:
            points: [pxo], [[n,3],[n,c],[b]]
            skip_points: [pxo], [[ns,3],[ns,c],[b]]
            cluster: [ns]

        Returns:
            points: [pxo], [[ns,3],[ns,c],[b]]
        """
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points

        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            # 必须在 FP32 下执行 interpolation, 防止 1/dist 在 amp 下出现 nan/inf
            proj_feat = self.proj(feat)
            import torch
            with torch.amp.autocast('cuda', enabled=False):
                feat = pointops.interpolation(
                    coord.float(), skip_coord.float(), proj_feat.float(), offset, skip_offset
                )

        if self.skip:
            feat = feat + self.proj_skip(skip_feat)

        return [skip_coord, feat, skip_offset]


class Encoder(nn.Module):
    """
    DeepLANet V2 Encoder, 先进行格网池化, 再进行BlockSequence处理

    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        grid_size: 体素大小
        neighbours: 邻域大小
        drop_path_rate: DropPath比例
        enable_checkpoint: checkpoint机制
        enable_layer_scale: 是否启用 LayerScale
        layer_scale_init_value: LayerScale 初始值
    """

    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        grid_size=None,
        neighbours=16,
        drop_path_rate=None,
        enable_checkpoint=False,
        enable_layer_scale=False,
        layer_scale_init_value=1e-5,
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
            enable_checkpoint=enable_checkpoint,
            enable_layer_scale=enable_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
        )

    def forward(self, points):
        """
        Args:
            points: [pxo], [[n,3],[n,c],[b]]

        Returns:
            points: [pxo], [[ns,3],[ns,c],[b]]
            cluster: [n]
        """
        points, cluster = self.down(points)
        return self.blocks(points), cluster


class Decoder(nn.Module):
    """
    DeepLANet V2 Decoder, 先进行上采样, 再进行BlockSequence处理

    Args:
        in_channels: 输入维度
        skip_channels: 跳跃连接维度
        embed_channels: 输出维度
        depth: 解码器深度
        neighbours: 邻域大小
        drop_path_rate: DropPath比例
        enable_checkpoint: checkpoint机制
        unpool_backend: 上采样方式
        enable_layer_scale: 是否启用 LayerScale
        layer_scale_init_value: LayerScale 初始值
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        depth,
        neighbours=16,
        drop_path_rate=None,
        enable_checkpoint=False,
        unpool_backend="map",
        enable_layer_scale=False,
        layer_scale_init_value=1e-5,
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
            enable_checkpoint=enable_checkpoint,
            enable_layer_scale=enable_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
        )

    def forward(self, points, skip_points, cluster):
        """
        Args:
            points: [pxo], [[ns,3],[ns,c],[b]]
            skip_points: [pxo], [[n,3],[n,c],[b]]
            cluster: [n]

        Returns:
            points: [pxo], [[n,3],[n,c],[b]]
        """
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class VFRPatchEmbed(nn.Module):
    """
    Patch Embedding for DeepLANet V2 using VFRModule
    基于 Stage-Level Position Embedding 优化

    Args:
        depth: 编码器深度
        in_channels: 输入维度
        embed_channels: 输出维度
        neighbours: 邻域大小
        drop_path_rate: DropPath比例
        enable_checkpoint: checkpoint机制
        enable_layer_scale: 是否启用 LayerScale
        layer_scale_init_value: LayerScale 初始值
    """

    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        neighbours=16,
        drop_path_rate=0.0,
        enable_checkpoint=False,
        enable_layer_scale=False,
        layer_scale_init_value=1e-5,
    ):
        super(VFRPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.neighbours = neighbours

        # 初始特征投影: in_channels -> embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        # PE encoder
        self.pe_encoder = PositionalEncodingEncoder(embed_channels)

        # VFR 嵌入
        self.vfr = VFRModule()

        # Block序列
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            neighbours=neighbours,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
            enable_layer_scale=enable_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
        )

    def forward(self, points):
        """
        Stage-Level Position Embedding:
        在初始嵌入时计算一次位置编码

        Args:
            points: [pxo], [[n,3],[n,c],[b]]

        Returns:
            points: [pxo], [[n,3],[n,c],[b]]
        """
        coord, feat, offset = points

        # 特征投影
        feat = self.proj(feat)  # [n, embed_channels]

        # Stage级别: 计算一次 KNN 查询和位置编码
        with torch.no_grad():
            reference_index, dist = pointops.knn_query(self.neighbours, coord, offset)
            # 计算原始位置编码: [n, k, 10]
            pos_encoding = compute_stage_positional_encoding(coord, reference_index, dist)

        # 编码位置特征: [n, k, 10] -> [n, C]
        pe = self.pe_encoder(pos_encoding)

        # VFR 嵌入: 融合 PE 后进行局部特征聚合
        feat = feat + pe
        feat = self.vfr(feat, coord, reference_index)

        return self.blocks([coord, feat, offset])


@MODELS.register_module("DeepLANet-v2")
@MODELS.register_module("DeepLANet-V2")
class DeepLANetV2(PointModule):
    """
    DeepLANet V2 Backbone

    核心改进:
    1. Stage-Level Position Embedding (与 V1 一致)
    2. VFRModule 替代 LocalFeatureAggregation: 更轻量，仅边缘特征+max pooling
    3. ResLFEBlock: Front-Linear + PE加法融合 + VFR，维度保持不变
    4. 计算效率更高，显存占用更低
    5. 深层网络优化: 零初始化 + LayerScale 提升训练稳定性
    6. 混合深监督: 返回中间特征用于辅助损失计算

    作为纯 backbone 使用，不包含 seg_head。
    输入: data_dict (dict) 或 Point 对象，需包含 coord, feat, offset。
    输出: Point 对象，feat 为解码器最终输出特征，维度 dec_channels[0]。
           如果启用深监督，Point.aux_outputs 包含中间特征列表用于辅助损失计算。

    Args:
        in_channels: 输入特征维度
        patch_embed_depth: Patch Embedding深度
        patch_embed_channels: Patch Embedding输出维度
        patch_embed_neighbours: Patch Embedding邻域大小
        enc_depths: 编码器深度
        enc_channels: 编码器输出维度
        enc_neighbours: 编码器邻域大小
        dec_depths: 解码器深度
        dec_channels: 解码器输出维度
        dec_neighbours: 解码器邻域大小
        grid_sizes: 体素大小
        drop_path_rate: DropPath比例
        enable_checkpoint: checkpoint机制
        unpool_backend: 上采样方式
        enable_layer_scale: 是否启用 LayerScale (深层网络建议开启)
        layer_scale_init_value: LayerScale 初始值 (默认 1e-5，120层建议 1e-6)
        enable_deep_supervision: 是否启用混合深监督 (返回中间特征用于辅助损失)
    """

    def __init__(
        self,
        in_channels,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.12, 0.24, 0.48),
        drop_path_rate=0,
        enable_checkpoint=False,
        unpool_backend="map",
        enable_layer_scale=False,
        layer_scale_init_value=1e-5,
        enable_deep_supervision=False,
    ):
        super(DeepLANetV2, self).__init__()
        self.in_channels = in_channels
        self.num_stages = len(enc_depths)
        self.enable_deep_supervision = enable_deep_supervision

        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)

        # 点云嵌入层
        self.patch_embed = VFRPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            enable_checkpoint=enable_checkpoint,
            enable_layer_scale=enable_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
        )

        # drop率逐渐提高
        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        dec_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))
        ]

        # 前一层的输出维度作为下一层的输入维度
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]

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
                    sum(enc_depths[:i]): sum(enc_depths[:i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
                enable_layer_scale=enable_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                neighbours=dec_neighbours[i],
                drop_path_rate=dec_dp_rates[
                    sum(dec_depths[:i]): sum(dec_depths[:i + 1])
                ],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend,
                enable_layer_scale=enable_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)

    def forward(self, data_dict):
        """
        Args:
            data_dict (dict 或 Point): 需包含 "coord" [n, 3], "feat" [n, c], "offset" [b]

        Returns:
            Point 对象, feat 为解码器最终输出特征 [n, dec_channels[0]]
            如果启用深监督，Point.aux_outputs 包含中间特征列表用于辅助损失计算
        """
        # 兼容 dict 和 Point 两种输入
        if not isinstance(data_dict, Point):
            point = Point(data_dict)
        else:
            point = data_dict

        coord = point.coord
        feat = point.feat
        offset = point.offset.int()

        points = [coord, feat, offset]
        points = self.patch_embed(points)

        # 用于深监督的中间特征
        aux_outputs = []

        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)
            skips.append([points])

            # 如果启用深监督，保存每个 encoder stage 的输出
            if self.enable_deep_supervision:
                aux_outputs.append(points)

        points = skips.pop(-1)[0]
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)

        coord, feat, offset = points

        point.feat = feat

        # 如果启用深监督，将中间特征附加到 Point 对象
        if self.enable_deep_supervision:
            point.aux_outputs = aux_outputs

        return point
