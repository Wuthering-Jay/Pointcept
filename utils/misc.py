import json

def extract_sorted_weights(file_path):
    # 1. 读取 JSON 文件
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. 获取 weights 字典
    weights = data.get("weights", {})
    
    # 3. 提取所有 id 并转换为整数，然后排序
    sorted_ids = sorted(int(id_str) for id_str in weights.keys())
    
    # 4. 按排序后的 id 顺序提取权重值
    weights_list = [weights[str(id)] for id in sorted_ids]
    
    return weights_list

def extract_label_id(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有键并转为列表
    keys_list = sorted(data.keys(), key=lambda x: int(x))
    
    return keys_list