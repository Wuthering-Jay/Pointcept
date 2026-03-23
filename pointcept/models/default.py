import torch
import torch.nn as nn
import torch_scatter
import torch_cluster

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    # 基础分割器模型
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        # 接收 backbone 和 criteria 的配置并构建实例
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        # 将输入字典传入 backbone 进行前向传播
        seg_logits = self.backbone(input_dict)
        # train, 计算损失
        if self.training:
            loss = self.criteria(seg_logits, input_dict)
            return dict(loss=loss)
        # eval，存在标签"segment"时，计算损失和 logits
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict)
            return dict(loss=loss, seg_logits=seg_logits)
        # test, 计算分割 logits
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    # 分割器模型 V2，支持更复杂的网络结构和特征聚合
    # 该模型在 v1.5.0 版本中引入，支持 Point 对象作为输入和输出
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        # 建立一个线性层作为分割头
        # 如果 num_classes 大于 0，则使用 nn.Linear，否则使用 nn.Identity 保持不变
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        # 冻结 backbone 的部分参数
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        # 将输入字典传入包装为 Point 对象
        point = Point(input_dict)
        # 部分网络支持 Point 对象作为输入和输出：pv3、sonata
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        # 若返回 Point 对象，循环处理可能的池化/上采样层级结构，进行特征聚合
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        # 特征传入分割头
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        # 如果需要返回 Point 对象，添加到返回字典中
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train，计算损失
        if self.training:
            loss = self.criteria(seg_logits, input_dict)
            return_dict["loss"] = loss
        # eval，存在标签"segment"时，计算损失和 logits
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict)
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test，计算分割 logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    # DINO 增强分割器模型，支持 DINO 特征和坐标的输入
    # DINO 是一种自监督学习方法，增强了模型的特征表示能力
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict)
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict)
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    # 分类器模型
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        # 分类头，包含多个线性层和 BN、ReLU 、Dropout 层
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        # 将输入字典传入包装为 Point 对象
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        # 基于 offset 进行平均池化特征聚合
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict)
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict)
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
        

@MODELS.register_module()
class DeepLASegmentor(nn.Module):
    """DeepLANet 专用 Segmentor，支持混合深监督 (Hybrid Deep Supervision)。

    基于 DefaultSegmentorV2，增加以下功能:
    1. 收集 backbone 的 aux_outputs (各 Encoder 阶段的中间特征)
    2. 使用 pointops.interpolation 将低分辨率中间特征插值到原始分辨率
    3. 动态辅助头 (Auxiliary Heads): 为每个 Encoder 阶段生成带 Dropout 的分类头
    4. 双轨损失机制: criteria (主损失) + aux_criteria (辅助损失)

    Args:
        num_classes (int): 分类类别数
        backbone_out_channels (int): backbone 输出特征维度
        backbone (dict): backbone 配置
        criteria (list[dict]): 主损失配置 (如 CE + Lovasz)
        aux_criteria (list[dict]): 辅助损失配置 (如纯 CE)
        aux_channels (list[int]): 各 Encoder 阶段的输出通道数，用于创建辅助头
        aux_dropout (float): 辅助头的 Dropout 概率
        aux_weights (tuple[float]): 各 stage 的辅助损失权重比例，如 (0.1, 0.2, 0.3, 0.4)
                                     如果为 None，默认所有 stage 权重相等 (1.0)
        freeze_backbone (bool): 是否冻结 backbone 参数
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        aux_criteria=None,
        aux_channels=None,
        aux_dropout=0.1,
        aux_weights=None,
        freeze_backbone=False,
    ):
        super().__init__()
        import pointops

        self.pointops = pointops
        self.num_classes = num_classes

        # 主分割头
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        # Backbone
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.aux_criteria = build_criteria(aux_criteria) if aux_criteria else None

        # 冻结 backbone
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 动态创建辅助头
        # aux_channels 应与 backbone 的 enc_channels 对应
        self.aux_heads = None
        self.aux_weights = None
        if aux_channels is not None and len(aux_channels) > 0:
            self.aux_heads = nn.ModuleList()
            for ch in aux_channels:
                head = nn.Sequential(
                    nn.Dropout(p=aux_dropout),
                    nn.Linear(ch, num_classes),
                )
                self.aux_heads.append(head)

            # 设置各 stage 的权重比例
            if aux_weights is not None:
                assert len(aux_weights) == len(aux_channels), (
                    f"aux_weights 长度 ({len(aux_weights)}) 必须与 "
                    f"aux_channels 长度 ({len(aux_channels)}) 一致"
                )
                self.aux_weights = aux_weights
            else:
                # 默认权重均为 1.0
                self.aux_weights = tuple([1.0] * len(aux_channels))

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)

        # 保存原始坐标和 offset，用于辅助特征插值
        origin_coord = point.coord.clone()
        origin_offset = point.offset.clone()

        # Backbone 前向
        point = self.backbone(point)

        # 处理 pooling_parent 层级 (与 DefaultSegmentorV2 一致)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        # 主分割头
        seg_logits = self.seg_head(feat)

        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        # =====================================================================
        # 混合深监督: 处理辅助输出
        # =====================================================================
        aux_logits_list = []
        if (
            self.training
            and self.aux_heads is not None
            and self.aux_criteria is not None
            and hasattr(point, "aux_outputs")
            and point.aux_outputs is not None
        ):
            aux_outputs = point.aux_outputs

            for i, aux_points in enumerate(aux_outputs):
                if i >= len(self.aux_heads):
                    break

                aux_coord, aux_feat, aux_offset = aux_points

                # 使用 pointops.interpolation 将低分辨率特征插值到原始分辨率
                # interpolation(src_coord, dst_coord, src_feat, src_offset, dst_offset)
                # 必须在 FP32 下执行，否则 1/dist 的操作在 float16 (AMP) 下极易产生 inf/nan
                with torch.amp.autocast('cuda', enabled=False):
                    interpolated_feat = self.pointops.interpolation(
                        aux_coord.float(), origin_coord.float(), aux_feat.float(), aux_offset, origin_offset
                    )

                # 通过辅助头生成 logits
                aux_logits = self.aux_heads[i](interpolated_feat)
                aux_logits_list.append(aux_logits)

        # =====================================================================
        # 计算损失
        # =====================================================================
        if self.training:
            # 主损失
            loss = self.criteria(seg_logits, input_dict)

            # 辅助损失
            if len(aux_logits_list) > 0 and self.aux_criteria is not None:
                aux_loss = 0.0
                for i, aux_logits in enumerate(aux_logits_list):
                    stage_loss = self.aux_criteria(aux_logits, input_dict)
                    # 应用各 stage 的权重比例
                    weighted_loss = stage_loss * self.aux_weights[i]
                    aux_loss = aux_loss + weighted_loss
                # aux_criteria 的 loss_weight 已在 build_criteria 中处理
                loss = loss + aux_loss

            return_dict["loss"] = loss
            return_dict["l_aux"] = aux_loss if len(aux_logits_list) > 0 else None

        elif "segment" in input_dict.keys():
            # Eval 模式: 只计算主损失
            loss = self.criteria(seg_logits, input_dict)
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits

        else:
            # Test 模式
            return_dict["seg_logits"] = seg_logits

        return return_dict
