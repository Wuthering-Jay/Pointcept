# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
#
# # 检查torch是否可用
# print(f"PyTorch版本: {torch.__version__}")
# print(f"CUDA是否可用: {torch.cuda.is_available()}")
# print(f"当前设备: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
#
# # 创建一些简单的训练数据
# X = torch.randn(100, 10)  # 100个样本，每个样本10个特征
# y = torch.randint(0, 2, (100,)).float()  # 二分类标签
#
# # 转换为数据集和数据加载器
# dataset = TensorDataset(X, y)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
#
#
# # 定义一个简单的神经网络
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(10, 5)
#         self.fc2 = nn.Linear(5, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return self.sigmoid(x).squeeze()
#
#
# # 初始化模型、损失函数和优化器
# model = SimpleNN()
# criterion = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
#
# # 训练循环
# num_epochs = 5
# for epoch in range(num_epochs):
#     for batch_X, batch_y in dataloader:
#         # 前向传播
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# print("训练完成！PyTorch工作正常。")
#
# # 测试模型
# with torch.no_grad():
#     test_input = torch.randn(1, 10)
#     prediction = model(test_input)
#     print(f"测试输入预测结果: {prediction.item():.4f}")
#
# # 检查 CUDA 是否可用
# assert torch.cuda.is_available(), "CUDA 不可用！"
# device = torch.device('cuda')
#
# print(f"PyTorch 版本: {torch.__version__}")
# print(f"当前设备: {device}")
# print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
#
# from torch_sparse import SparseTensor
#
# print("\n验证 torch_sparse (GPU)...")
#
# # 创建稀疏矩阵并移到 GPU
# indices = torch.tensor([[0, 1, 1], [2, 0, 2]], device=device)  # GPU 坐标
# values = torch.tensor([3.0, 4.0, 5.0], device=device)  # GPU 数据
# sparse_tensor = SparseTensor(
#     row=indices[0],
#     col=indices[1],
#     value=values,
#     sparse_sizes=(3, 3)
# ).to(device)
#
# print("稀疏矩阵 (GPU):")
# print(sparse_tensor.to_dense())  # 应在 GPU 上输出稠密矩阵
#
# from torch_sparse import SparseTensor
#
# print("\n验证 torch_sparse (GPU)...")
#
# # 创建稀疏矩阵并移到 GPU
# indices = torch.tensor([[0, 1, 1], [2, 0, 2]], device=device)  # GPU 坐标
# values = torch.tensor([3.0, 4.0, 5.0], device=device)  # GPU 数据
# sparse_tensor = SparseTensor(
#     row=indices[0],
#     col=indices[1],
#     value=values,
#     sparse_sizes=(3, 3)
# ).to(device)
#
# print("稀疏矩阵 (GPU):")
# print(sparse_tensor.to_dense())  # 应在 GPU 上输出稠密矩阵
#
# from torch_cluster import radius_graph
#
# print("\n验证 torch_cluster (GPU)...")
#
# # 生成 GPU 上的随机点
# pos = torch.rand((5, 2), device=device)  # GPU 数据
# edge_index = radius_graph(pos, r=0.5, loop=False)  # 直接在 GPU 上计算
#
# print("点坐标 (GPU):")
# print(pos.cpu())  # 转回 CPU 打印（避免终端混叠）
# print("半径图边索引 (GPU):")
# print(edge_index.cpu())
#
# from torch_scatter import scatter_max
#
# print("\n验证 torch_scatter (GPU)...")
#
# # GPU 数据
# src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
# index = torch.tensor([0, 0, 1, 1, 2], device=device)
#
# # 计算 GPU 上的分组最大值
# out, _ = scatter_max(src, index, dim=-1)
#
# print("输入数据 (GPU):", src.cpu())
# print("分组索引 (GPU):", index.cpu())
# print("分组最大值 (GPU):", out.cpu())
#
#
#
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
#
# print("\n验证 torch_geometric (GPU)...")
#
# # 创建 GPU 上的图数据
# edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device=device)
# x = torch.randn((3, 4), device=device)
# data = Data(x=x, edge_index=edge_index).to(device)
#
# # 定义 GPU 上的 GCN 模型
# class GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = GCNConv(4, 2).to(device)  # 确保模型在 GPU 上
#
#     def forward(self, data):
#         return self.conv(data.x, data.edge_index)
#
# model = GCN().to(device)
# out = model(data)
#
# print("节点特征 (GPU):", x.cpu())
# print("GCN 输出 (GPU):", out.cpu())
#
# import pointops
# import einops
#
# class PointBatchNorm(nn.Module):
#     """
#     Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
#     对形状为[n, c], [n, l, c]的点云数据进行批量归一化
#     """
#
#     def __init__(self, embed_channels):
#         super().__init__()
#         self.norm = nn.BatchNorm1d(embed_channels)
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         if input.dim() == 3:
#             return (
#                 self.norm(input.transpose(1, 2).contiguous())
#                 .transpose(1, 2)
#                 .contiguous()
#             )
#         elif input.dim() == 2:
#             return self.norm(input)
#         else:
#             raise NotImplementedError
#
#
# class GroupedVectorAttention(nn.Module):
#     """
#     分组向量注意力机制
#     Args:
#         embed_channesl: 输入输出维度
#         groups: 分组数量
#         attn_drop_rate: drop比例
#         qkv_bias: 无用
#         pe_multiplier: 位置编码乘性因子
#         pe_bias: 位置编码偏置因子
#     """
#     def __init__(
#         self,
#         embed_channels,
#         groups,
#         attn_drop_rate=0.0,
#         qkv_bias=True,
#         pe_multiplier=False,
#         pe_bias=True,
#     ):
#         super(GroupedVectorAttention, self).__init__()
#         self.embed_channels = embed_channels
#         self.groups = groups
#         assert embed_channels % groups == 0
#         self.attn_drop_rate = attn_drop_rate
#         self.qkv_bias = qkv_bias
#         self.pe_multiplier = pe_multiplier
#         self.pe_bias = pe_bias
#
#         self.linear_q = nn.Sequential(
#             nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
#             PointBatchNorm(embed_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.linear_k = nn.Sequential(
#             nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
#             PointBatchNorm(embed_channels),
#             nn.ReLU(inplace=True),
#         )
#
#         self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)
#
#         if self.pe_multiplier:
#             self.linear_p_multiplier = nn.Sequential(
#                 nn.Linear(3, embed_channels),
#                 PointBatchNorm(embed_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(embed_channels, embed_channels),
#             )
#         if self.pe_bias:
#             self.linear_p_bias = nn.Sequential(
#                 nn.Linear(3, embed_channels),
#                 PointBatchNorm(embed_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(embed_channels, embed_channels),
#             )
#         self.weight_encoding = nn.Sequential(
#             nn.Linear(embed_channels, groups),
#             PointBatchNorm(groups),
#             nn.ReLU(inplace=True),
#             nn.Linear(groups, groups),
#         )
#         self.softmax = nn.Softmax(dim=1)
#         self.attn_drop = nn.Dropout(attn_drop_rate)
#
#     def forward(self, feat, coord, reference_index):
#         """
#         input: feat: [n, c], coord: [n, 3], reference_index: [n, k]
#         output: feat: [n, c]
#         """
#         query, key, value = (
#             self.linear_q(feat), # [n, c]
#             self.linear_k(feat), # [n, c]
#             self.linear_v(feat), # [n, c]
#         )
#         key = pointops.grouping(reference_index, key, coord, with_xyz=True) # [n, k, 3+c]
#         value = pointops.grouping(reference_index, value, coord, with_xyz=False) # [n, k, c]
#         pos, key = key[:, :, 0:3], key[:, :, 3:] # [n, k, 3], [n, k, c]
#         relation_qk = key - query.unsqueeze(1) # [n, k ,c], 邻域内与中心点的相对位置, 用于相对位置编码
#         if self.pe_multiplier: # 乘性因子
#             pem = self.linear_p_multiplier(pos) # [n, k, c]
#             relation_qk = relation_qk * pem # [n, k, c]
#         if self.pe_bias: # 偏置因子
#             peb = self.linear_p_bias(pos) # [n, k, c]
#             relation_qk = relation_qk + peb # [n, k, c]
#             value = value + peb # [n, k, c]
#
#         weight = self.weight_encoding(relation_qk) # [n, k, g]
#         weight = self.attn_drop(self.softmax(weight)) # [n, k, g]
#
#         mask = torch.sign(reference_index + 1) # [n, k], 无效邻域点标记为0
#         weight = torch.einsum("n s g, n s -> n s g", weight, mask) # [n, k, g]
#         value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups) # [n, k, g, i]
#         feat = torch.einsum("n s g i, n s g -> n g i", value, weight) # [n, g, i]
#         feat = einops.rearrange(feat, "n g i -> n (g i)") # [n, c]
#         return feat # [n, c]
#
#
# # 创建一个 GroupedVectorAttention 实例
# embed_channels = 64
# groups = 8
# attn_drop_rate = 0.1
# qkv_bias = True
# pe_multiplier = True
# pe_bias = True
#
# grouped_vector_attention = GroupedVectorAttention(
#     embed_channels=embed_channels,
#     groups=groups,
#     attn_drop_rate=attn_drop_rate,
#     qkv_bias=qkv_bias,
#     pe_multiplier=pe_multiplier,
#     pe_bias=pe_bias
# ).cuda()
#
# # 创建随机输入数据并移动到 GPU
# n = 100  # 点的数量
# c = embed_channels  # 特征维度
# k = 16  # 每个点的邻居数量
#
# feat = torch.randn(n, c).cuda()
# coord = torch.randn(n, 3).cuda()
# reference_index = torch.randint(0, n, (n, k)).cuda()
#
# # 调用 forward 方法
# output_feat = grouped_vector_attention(feat, coord, reference_index)
#
# # 打印输出特征的形状以验证结果
# print(f"Output feature shape: {output_feat.shape}")


# import numpy as np
# import laspy


# def npy_to_las(npy_path, las_path):
#     """
#     将存储n*3坐标的NPY文件转换为LAS点云文件

#     参数:
#         npy_path: 输入的NPY文件路径
#         las_path: 输出的LAS文件路径
#     """
#     # 读取NPY文件
#     coords = np.load(npy_path)

#     # 验证数据形状
#     if coords.shape[1] != 3:
#         raise ValueError("输入数据必须是n*3的坐标数组")

#     # 创建LAS文件头
#     header = laspy.LasHeader(point_format=0, version="1.2")

#     # 创建LAS文件
#     las = laspy.LasData(header)

#     # 设置坐标
#     las.x = coords[:, 0]
#     las.y = coords[:, 1]
#     las.z = coords[:, 2]

#     # 保存LAS文件
#     las.write(las_path)
#     print(f"成功将 {npy_path} 转换为 {las_path}")


# # 使用示例
# npy_file = r"D:\data\天津样例数据\细粒度8\npy\testsa\8平原分类前1sa_segment_000\coord.npy"  # 输入的NPY文件，需要指定为npy文件夹下的coord.npy文件
# las_file = r"D:\data\天津样例数据\细粒度8\npy\output.las"  # 输出的LAS文件
# npy_to_las(npy_file, las_file)
import numpy as np
import laspy
from tqdm import tqdm
from scipy.spatial import cKDTree
import time

def modify_class_by_proximity(input_las_path, output_las_path, distance_threshold=0.05):
    """
    使用KD树优化的方法修改距离类别17点很近的类别1点
    
    Args:
        input_las_path (str): 输入LAS文件路径
        output_las_path (str): 输出LAS文件路径
        distance_threshold (float): 距离阈值（米），默认0.05m
    """
    class_=2
    print("Reading LAS file...")
    start_time = time.time()
    
    # 读取LAS文件
    las = laspy.read(input_las_path)
    
    # 获取点坐标
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # 获取分类数组
    classification = las.classification
    
    # 找到类别1和类别17的点的索引
    class_1_indices = np.where(classification == 1)[0]
    class_17_indices = np.where(classification == class_)[0]
    
    print(f"Found {len(class_1_indices)} points in class 1")
    print(f"Found {len(class_17_indices)} points in class 17")
    
    if len(class_17_indices) == 0:
        print("No class 17 points found in the file.")
        return
    
    # 获取类别1和类别17的点坐标
    class_1_points = points[class_1_indices]
    class_17_points = points[class_17_indices]
    
    print("\nBuilding KD-Tree...")
    # 构建KD树
    tree = cKDTree(class_17_points)
    
    print("Finding nearest neighbors...")
    # 使用KD树查询最近邻+
    # 返回到最近点的距离
    distances, _ = tree.query(class_1_points, k=1)
    
    # 找到距离在阈值内的点
    points_to_modify_mask = distances <= distance_threshold
    points_to_modify = class_1_indices[points_to_modify_mask]
    
    # 修改分类
    classification[points_to_modify] = class_
    
    print("\nCreating output LAS file...")
    # 创建输出LAS文件
    output_las = laspy.LasData(las.header)
    output_las.points = las.points
    output_las.classification = classification
    
    # 保存修改后的LAS文件
    print("Saving modified LAS file...")
    output_las.write(output_las_path)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nProcessing complete:")
    print(f"- Modified {len(points_to_modify)} points from class 1 to class 17")
    print(f"- Processing time: {processing_time:.2f} seconds")
    print(f"- Output saved to: {output_las_path}")

# 示例使用
if __name__ == "__main__":
    input_file = r"E:\data\连云港\连云港-new.las"
    output_file = r"E:\data\连云港\连云港-new.las"
    
    print("=== LAS Point Cloud Classification Modifier ===")
    print(f"Start time: 2025-06-11 09:43:00 UTC")
    print(f"User: Wuthering-Jay")
    print(f"Processing file: {input_file}")
    print("-" * 45)
    
    modify_class_by_proximity(input_file, output_file, distance_threshold=1)