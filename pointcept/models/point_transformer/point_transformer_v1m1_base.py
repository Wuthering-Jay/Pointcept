"""
PointNext mode 1 for Semantic Segmentation
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
