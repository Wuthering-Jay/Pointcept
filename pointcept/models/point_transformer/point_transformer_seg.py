"""
Point Transformer V1 for Semantic Segmentation

Might be a bit different from the original paper

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import einops
import pointops

from pointcept.models.builder import MODELS
from .utils import LayerNorm1d


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
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
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
        share_planes: 共享特征维度
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


class PointTransformerSeg(nn.Module):
    """
    Point Transformer for Semantic Segmentation
    Args:
        block: block type, 默认为Bottleneck
        blocks: block的数量
        in_channels: 输入特征维度
        num_classes: 输出类别数
    """
    def __init__(self, block, blocks, in_channels=6, num_classes=13):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(
            block,
            planes[0],
            blocks[0],
            share_planes,
            stride=stride[0],
            nsample=nsample[0],
        )  # N/1
        self.enc2 = self._make_enc(
            block,
            planes[1],
            blocks[1],
            share_planes,
            stride=stride[1],
            nsample=nsample[1],
        )  # N/4
        self.enc3 = self._make_enc(
            block,
            planes[2],
            blocks[2],
            share_planes,
            stride=stride[2],
            nsample=nsample[2],
        )  # N/16
        self.enc4 = self._make_enc(
            block,
            planes[3],
            blocks[3],
            share_planes,
            stride=stride[3],
            nsample=nsample[3],
        )  # N/64
        self.enc5 = self._make_enc(
            block,
            planes[4],
            blocks[4],
            share_planes,
            stride=stride[4],
            nsample=nsample[4],
        )  # N/256
        self.dec5 = self._make_dec(
            block, planes[4], 1, share_planes, nsample=nsample[4], is_head=True
        )  # transform p5
        self.dec4 = self._make_dec(
            block, planes[3], 1, share_planes, nsample=nsample[3]
        )  # fusion p5 and p4
        self.dec3 = self._make_dec(
            block, planes[2], 1, share_planes, nsample=nsample[2]
        )  # fusion p4 and p3
        self.dec2 = self._make_dec(
            block, planes[1], 1, share_planes, nsample=nsample[1]
        )  # fusion p3 and p2
        self.dec1 = self._make_dec(
            block, planes[0], 1, share_planes, nsample=nsample[0]
        )  # fusion p2 and p1
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[0], num_classes),
        )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        """
        Args:
            block: block type, 默认为Bottleneck
            planes: 输出特征维度
            blocks: block的数量
            share_planes: 共享特征维度
            stride: 降采样率
            nsample: 每个点采样的点数
        """
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def _make_dec(
        self, block, planes, blocks, share_planes=8, nsample=16, is_head=False
    ):
        """
        Args:
            block: block type, 默认为Bottleneck
            planes: 输出特征维度
            blocks: block的数量
            share_planes: 共享特征维度
            nsample: 每个点采样的点数
            is_head: 是否为head
        """
        layers = [
            TransitionUp(self.in_planes, None if is_head else planes * block.expansion)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, data_dict):
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1]) 
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x


@MODELS.register_module("PointTransformer-Seg26")
class PointTransformerSeg26(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg26, self).__init__(
            Bottleneck, [1, 1, 1, 1, 1], **kwargs
        )


@MODELS.register_module("PointTransformer-Seg38")
class PointTransformerSeg38(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg38, self).__init__(
            Bottleneck, [1, 2, 2, 2, 2], **kwargs
        )


@MODELS.register_module("PointTransformer-Seg50")
class PointTransformerSeg50(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg50, self).__init__(
            Bottleneck, [1, 2, 3, 5, 2], **kwargs
        )
