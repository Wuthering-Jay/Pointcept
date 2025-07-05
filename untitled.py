# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import laspy

# # --- 1. 配置参数 ---
# # 请将这些文件名替换为你的实际文件名
# CSV_FILE_PATH = r"E:\data\STPLS3D\RealWorldData\usc\USC_points - Cloud-1_统计单木属性.csv" # 你的CSV文件路径
# INPUT_LAS_PATH = r"E:\data\STPLS3D\RealWorldData\usc\USC_points - Cloud-1_按树ID提取点云.las"             # 你的原始LAS文件路径
# OUTPUT_LAS_PATH = r"E:\data\STPLS3D\RealWorldData\usc\USC_points-result.las" # 你希望保存的输出LAS文件路径

# # 用于聚类的特征列名
# FEATURES = ['TreeHeight', 'CrownDiameter', 'CrownArea', 'CrownVolume']

# # 新分类ID的起始编号 (按要求从21开始)
# CLASSIFICATION_START_ID = 21

# # --- 2. 加载和预处理CSV数据 ---
# print("步骤 2: 加载并准备CSV数据...")
# try:
#     # 读取CSV文件
#     df = pd.read_csv(CSV_FILE_PATH)
    
#     # 确保TreeID是整数类型，并将其设置为索引
#     df['TreeID'] = df['TreeID'].astype(int)
#     df.set_index('TreeID', inplace=True)

#     # 提取用于聚类的特征
#     X = df[FEATURES]
#     print(f"成功加载 {len(df)} 棵树的数据。")
#     print("用于聚类的特征:", FEATURES)

# except FileNotFoundError:
#     print(f"错误: 找不到CSV文件 '{CSV_FILE_PATH}'。请检查路径是否正确。")
#     exit()
# except KeyError as e:
#     print(f"错误: CSV文件中缺少必要的列: {e}。请确保文件包含 {FEATURES} 和 'TreeID'。")
#     exit()

# # 数据标准化（非常重要！）
# # K-Means等基于距离的算法对特征的尺度非常敏感。
# # 例如，CrownVolume的数值远大于TreeHeight，如果不进行标准化，聚类结果将主要由CrownVolume决定。
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # --- 3. 确定最佳聚类数 (K) ---
# # print("\n步骤 3: 使用肘部法则确定最佳聚类数 (K)...")
# # inertia = []
# # k_range = range(2, 11) # 测试K从2到10

# # for k in k_range:
# #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
# #     kmeans.fit(X_scaled)
# #     inertia.append(kmeans.inertia_)

# # # 绘制肘部图
# # plt.figure(figsize=(10, 6))
# # plt.plot(k_range, inertia, 'bo-')
# # plt.xlabel('聚类数量 (K)')
# # plt.ylabel('簇内误差平方和 (Inertia)')
# # plt.title('肘部法则确定最佳K值')
# # plt.xticks(k_range)
# # plt.grid(True)
# # plt.show()

# # print("请观察上图中的“肘部”。")
# # print("“肘部”是曲线斜率急剧变化的那个点，通常是最佳的K值。")
# # # 等待用户输入他们选择的K值
# # try:
# #     OPTIMAL_K = int(input("根据上图，请输入你选择的最佳K值: "))
# # except ValueError:
# #     print("无效输入，请输入一个整数。")
# #     exit()
# OPTIMAL_K=4

# # --- 4. 执行K-Means聚类 ---
# print(f"\n步骤 4: 使用 K={OPTIMAL_K} 执行K-Means聚类...")
# kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
# clusters = kmeans.fit_predict(X_scaled)

# # 将聚类结果 (0, 1, 2...) 映射到新的分类ID (21, 22, 23...)
# df['Cluster'] = clusters
# df['NewClassification'] = df['Cluster'] + CLASSIFICATION_START_ID

# print("聚类完成。各簇的树木数量:")
# print(df['NewClassification'].value_counts().sort_index())

# # 创建一个从 TreeID 到 NewClassification 的映射字典
# # 这将是更新LAS文件的关键
# tree_id_to_classification_map = df['NewClassification'].to_dict()

# # --- 5. 更新LAS文件 (最终修正版，采用最稳健的“读-改-写”流程) ---
# print("\n步骤 5: 读取LAS文件并更新Classification字段...")
# try:
#     # 1. 使用 laspy.read() 一次性将整个文件读入内存
#     print(f"正在读取 '{INPUT_LAS_PATH}'...")
#     las = laspy.read(INPUT_LAS_PATH)
#     print(f"成功读取 {len(las.points)} 个点。")

#     # 2. 准备新的分类数据 (这部分逻辑不变，非常可靠)
#     # 获取LAS文件中的TreeID字段
#     las_tree_ids = None
#     possible_tree_id_fields = ['tree_id', 'Tree_ID', 'treeID']
#     for field in possible_tree_id_fields:
#         if field in las.header.point_format.dimension_names:
#             las_tree_ids = las[field]
#             print(f"成功在LAS文件中找到 '{field}' 字段。")
#             break
    
#     if las_tree_ids is None:
#         # 使用 raise 抛出错误，会中断程序并显示更清晰的信息
#         raise RuntimeError(f"错误: 在LAS文件的点格式中找不到TreeID字段。已尝试: {possible_tree_id_fields}")

#     # 使用pandas进行安全映射
#     las_df = pd.DataFrame({
#         'Original_Classification': las.classification,
#         'TreeID': las_tree_ids
#     })
#     classification_map = pd.Series(tree_id_to_classification_map)
#     las_df['New_Classification'] = las_df['TreeID'].map(classification_map)
#     las_df['Final_Classification'] = las_df['New_Classification'].fillna(las_df['Original_Classification'])
#     final_classifications = las_df['Final_Classification'].to_numpy().astype(np.uint8)

#     # 3. 在内存中直接修改 LAS 对象的 classification 字段
#     print("正在内存中更新点的分类...")
#     las.classification = final_classifications

#     # 4. 将修改后的完整 LAS 对象写入新文件
#     print(f"正在将结果写入新文件: '{OUTPUT_LAS_PATH}'...")
#     las.write(OUTPUT_LAS_PATH)

#     print(f"处理完成！更新后的LAS文件已保存到: '{OUTPUT_LAS_PATH}'")

# except FileNotFoundError:
#     print(f"错误: 找不到LAS文件 '{INPUT_LAS_PATH}'。请检查路径是否正确。")
#     exit()
# except Exception as e:
#     # 增加一个更详细的错误追踪，如果再出问题，可以提供更多信息
#     import traceback
#     print(f"处理LAS文件时发生未知错误: {e}")
#     traceback.print_exc()

import laspy
import numpy as np

# --- 1. 配置参数 ---
# 请将这些文件名替换为你的实际文件名
CLUSTERED_LAS_PATH = r"E:\data\STPLS3D\RealWorldData\usc\USC_points-result.las"  # 你之前聚类得到的、有TreeID字段的LAS文件
OTHER_LAS_PATH = r"E:\data\STPLS3D\RealWorldData\usc\USC_points - Cloud-2.las"     # 你要合并的、没有TreeID字段的LAS文件
MERGED_LAS_PATH = r"E:\data\STPLS3D\RealWorldData\usc\merged_output.las"        # 最终合并后保存的文件名

# --- 2. 读取两个LAS文件 ---
print("步骤 2: 读取输入文件...")
try:
    clustered_las = laspy.read(CLUSTERED_LAS_PATH)
    print(f"  - '{CLUSTERED_LAS_PATH}' 读取成功 (Offset: {clustered_las.header.offsets})")
    other_las = laspy.read(OTHER_LAS_PATH)
    print(f"  - '{OTHER_LAS_PATH}' 读取成功 (Offset: {other_las.header.offsets})")
except FileNotFoundError as e:
    print(f"错误: 找不到文件 {e.filename}。请检查路径是否正确。")
    exit()

# --- 3. 创建统一的头信息 (Header) ---
# 我们以 clustered_las 的头为基础，因为它包含了 TreeID
print("步骤 3: 创建统一的头信息...")
new_header = clustered_las.header
new_header.point_count = len(clustered_las.points) + len(other_las.points)
# 注意：Offset 和 Scale 将在最后由 laspy 根据合并后的点自动计算或我们手动设置，这里暂不处理

# --- 4. 合并点数据 (正确的方式) ---
print("步骤 4: 合并点数据...")

# 创建一个新的LasData对象来存放合并后的数据
merged_las = laspy.LasData(new_header)

# !! 关键修正：使用 .xyz 属性直接获取并合并真实世界坐标 !!
# laspy 会自动处理 Offset 和 Scale 的转换
print("  - 正在合并真实世界坐标 (X, Y, Z)...")
merged_xyz = np.concatenate((clustered_las.xyz, other_las.xyz))
merged_las.xyz = merged_xyz

# --- 合并其他维度 (这部分逻辑不变) ---
# 获取共有的维度名称 (除x,y,z外)
clustered_dims = set(clustered_las.point_format.dimension_names)
other_dims = set(other_las.point_format.dimension_names)
# 从共有维度中移除我们已手动处理的x,y,z
common_dims = list(clustered_dims.intersection(other_dims) - {'X', 'Y', 'Z'})

print(f"  - 正在合并共有的其他维度: {common_dims}")
for dim in common_dims:
    data1 = clustered_las[dim]
    data2 = other_las[dim]
    merged_las[dim] = np.concatenate((data1, data2))

# --- 特殊处理 TreeID 字段 (逻辑不变) ---
print("  - 正在处理 'TreeID' 维度...")
tree_id_dtype = clustered_las.Tree_ID.dtype
other_tree_ids = np.zeros(len(other_las.points), dtype=tree_id_dtype)
merged_las.Tree_ID = np.concatenate((clustered_las.Tree_ID, other_tree_ids))


# --- 5. 写入最终的LAS文件 ---
print("步骤 5: 写入合并后的文件...")
# 在写入时，laspy会根据所有点的坐标范围自动计算一个合适的新Offset和Scale
merged_las.write(MERGED_LAS_PATH)

print(f"\n合并成功！错位问题已解决。")
print(f"最终文件 '{MERGED_LAS_PATH}' 已保存，包含 {len(merged_las.points)} 个点。")