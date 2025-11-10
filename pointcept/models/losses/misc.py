"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, data_dict):
        target = data_dict.get("segment") if "segment" in data_dict else data_dict["category"]
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, data_dict):
        target = data_dict.get("segment") if "segment" in data_dict else data_dict["category"]
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, data_dict, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            data_dict (dict): Dictionary containing 'segment' or 'category' key with ground truth.
        Returns:
            torch.Tensor: The calculated loss
        """
        target = data_dict.get("segment") if "segment" in data_dict else data_dict["category"]
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, data_dict, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            data_dict (dict): Dictionary containing 'segment' or 'category' key with ground truth.
        Returns:
            torch.Tensor: The calculated loss
        """
        target = data_dict.get("segment") if "segment" in data_dict else data_dict["category"]
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        loss = loss.sum(dim=1)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, data_dict, **kwargs):
        target = data_dict.get("segment") if "segment" in data_dict else data_dict["category"]
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


@LOSSES.register_module()
class LocalConsistencyLoss(nn.Module):
    def __init__(self, k_neighbors=16, ignore_index=-1, loss_weight=1.0):
        """Local Consistency Loss for point cloud segmentation.
        This loss encourages neighboring points to have similar predictions.
        """
        super(LocalConsistencyLoss, self).__init__()
        self.k_neighbors = k_neighbors
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, pred, data_dict):
        # Lazy import to avoid conflicts in DataLoader workers
        import pointops
        
        # 1. 从 pred 和 data_dict 中提取所有需要的数据
        seg_logits = pred
        coord = data_dict["coord"]
        offset = data_dict["offset"]
        
        # 否则使用 __init__ 中配置的
        labels = data_dict["segment"] # labels 是可选的

        # --- 以下是 LocalConsistencyLoss 的完整逻辑 ---
        probs = F.softmax(seg_logits, dim=1) # [N, C]

        with torch.no_grad():
            reference_index, _ = pointops.knn_query(
                self.k_neighbors, coord, offset
            ) # [N, k]
            mask = torch.sign(reference_index + 1).float() # [N, k]
            
            valid_points_mask = None
            if labels is not None and self.ignore_index is not None:
                valid_points_mask = (labels != self.ignore_index) # [N]
                mask = mask * valid_points_mask.unsqueeze(1) # [N, k]

        neighbor_probs = pointops.grouping(
            reference_index, probs, coord, with_xyz=False
        ) # [N, k, C]

        center_probs = probs.unsqueeze(1) # [N, 1, C]
        diff_sq = (center_probs - neighbor_probs) ** 2
        dist_sq = diff_sq.sum(dim=2) # [N, k]
        
        masked_dist_sq = dist_sq * mask # [N, k]
        
        sum_dist_per_point = masked_dist_sq.sum(dim=1) # [N]
        num_valid_neighbors = mask.sum(dim=1).clamp(min=1.0) # [N]
        mean_dist_per_point = sum_dist_per_point / num_valid_neighbors # [N]
        
        if valid_points_mask is not None:
            total_loss = mean_dist_per_point[valid_points_mask].mean()
        else:
            total_loss = mean_dist_per_point.mean()
            
        return total_loss * self.loss_weight
    

@LOSSES.register_module()
class LACLoss(nn.Module):
    """
    标签感知一致性损失 (受超点/监督对比损失启发)
    
    核心思想: 
    只在那些“几何相近”且“真实语义标签相同”的点对之间, 
    强制要求它们的“预测概率”保持一致。
    
    这避免了在不同物体的边界处进行不必要的平滑，从而保护了边界的清晰度。
    """
    def __init__(self, k_neighbors=16, ignore_index=-1, loss_weight=1.0):
        """
        Args:
            k_neighbors (int): 用于计算一致性损失的邻居数量
            ignore_index (int, optional): 训练时要忽略的标签索引
            loss_weight (float): 此损失的权重
        """
        super(LACLoss, self).__init__()
        self.k_neighbors = k_neighbors
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        
    def forward(self, pred, data_dict):
        """
        Args:
            pred (torch.Tensor): 网络的原始输出 (seg_logits), shape [N, num_classes]
            data_dict (dict): 包含所有需要信息的字典, 必须包含:
                              - 'coord' (torch.Tensor): 点的坐标, shape [N, 3]
                              - 'offset' (torch.Tensor): 批次信息, shape [B]
                              - 'segment' (torch.Tensor): 真实标签, shape [N]
        Returns:
            torch.Tensor: 一个标量的标签感知平滑损失值
        """
        # Lazy import to avoid conflicts in DataLoader workers
        import pointops
        
        # 1. 从 pred 和 data_dict 中提取所有需要的数据
        seg_logits = pred
        coord = data_dict["coord"]
        offset = data_dict["offset"].int()
        # 根据您的代码，标签键名为 "segment"
        labels = data_dict["segment"] 
        
        # 2. 计算预测概率
        # 这个计算需要保留梯度，因此在 no_grad() 之外
        probs = F.softmax(seg_logits, dim=1) # [N, C]

        # 3. 寻找邻居并计算所有掩码 (不需要梯度)
        with torch.no_grad():
            # 3.1 寻找K近邻
            reference_index, _ = pointops.knn_query(
                self.k_neighbors, coord, offset
            ) # [N, k]
            
            # 3.2 创建有效的KNN掩码 (无效邻居索引为 -1)
            knn_mask = torch.sign(reference_index + 1).bool() # [N, k]

            # 3.3 分组邻居的真实标签
            # grouping 需要 [N, C] 格式, 所以先 unsqueeze
            neighbor_labels = pointops.grouping(
                reference_index, labels.unsqueeze(1).float(), coord, with_xyz=False
            ) # [N, k, 1]
            neighbor_labels = neighbor_labels.squeeze(-1).long() # [N, k]

            # 3.4 语义掩码: 中心点和邻居点是否具有相同的真实标签?
            # labels.unsqueeze(1) -> [N, 1]
            same_label_mask = (labels.unsqueeze(1) == neighbor_labels) # [N, k]

            # 3.5 忽略掩码: 中心点和邻居点是否都不是 ignore_index?
            center_valid_mask = (labels != self.ignore_index) # [N]
            neighbor_valid_mask = (neighbor_labels != self.ignore_index) # [N, k]
            
            # 3.6 最终掩码: 我们只关心那些...
            # (1) 是有效邻居 (knn_mask)
            # (2) 中心点不是 ignore (center_valid_mask)
            # (3) 邻居点不是 ignore (neighbor_valid_mask)
            # (4) 中心点和邻居点标签相同 (same_label_mask)
            total_mask = (
                knn_mask & 
                center_valid_mask.unsqueeze(1) & 
                neighbor_valid_mask & 
                same_label_mask
            ) # [N, k]

        # 4. 分组邻居的预测概率 (这步需要梯度)
        neighbor_probs = pointops.grouping(
            reference_index, probs, coord, with_xyz=False
        ) # [N, k, C]
        
        # 5. 计算损失
        
        # 5.1 计算中心点和邻居点的概率分布差异 (L2 距离的平方)
        # center_probs -> [N, 1, C]
        center_probs = probs.unsqueeze(1)
        # prob_dist_sq -> [N, k, C] -> [N, k]
        prob_dist_sq = ((center_probs - neighbor_probs)**2).sum(dim=2)

        # 5.2 应用最终的 "标签感知" 掩码
        masked_prob_dist_sq = prob_dist_sq * total_mask.float() # [N, k]
        
        # 5.3 计算平均损失 (只对所有有效的点对 "pairs" 求均值)
        sum_loss = masked_prob_dist_sq.sum()
        num_valid_pairs = total_mask.sum().clamp(min=1.0)
        
        mean_loss = sum_loss / num_valid_pairs
            
        return mean_loss * self.loss_weight