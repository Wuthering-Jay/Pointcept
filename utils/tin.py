import ctypes
import os
import time
import shutil
import tempfile
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm
import laspy
import numpy as np

# ================= 配置区域 =================
# DLL 路径 (固定值)
DLL_PATH = r"libs\Release\LiDAROprationDLLEx.dll"
DEN_DLL_PATH = r"libs\Release\den\LiDAROprationDLLEx.dll"
# DLL 支持的最大 LAS 版本
DLL_MAX_LAS_VERSION = (1, 2)
# ===========================================


def get_las_format_info(las_path: str) -> Dict[str, Any]:
    """
    获取 LAS 文件的格式信息，用于后续还原
    
    Returns:
        包含版本、点格式、header 信息的字典
    """
    las_data = laspy.read(las_path)
    
    info = {
        "version": (las_data.header.version.major, las_data.header.version.minor),
        "point_format": las_data.header.point_format.id,
        "offsets": las_data.header.offsets.copy(),
        "scales": las_data.header.scales.copy(),
        "vlrs": list(las_data.header.vlrs) if hasattr(las_data.header, 'vlrs') else [],
        "extra_dims": [(dim.name, dim.dtype, getattr(dim, 'description', '')) 
                       for dim in las_data.point_format.extra_dimensions],
    }
    
    del las_data
    return info


def convert_las_to_compatible_version(input_path: str, output_path: str, target_version: Tuple[int, int] = (1, 2)) -> Dict[str, Any]:
    """
    将 LAS 文件转换为兼容的版本格式（用于 DLL 处理）
    同时保存原始格式信息用于后续还原
    
    Args:
        input_path: 输入 LAS 文件路径
        output_path: 输出 LAS 文件路径
        target_version: 目标版本，默认 (1, 2)
    
    Returns:
        原始格式信息字典，如果无需转换返回 None
    """
    las_data = laspy.read(input_path)
    current_version = (las_data.header.version.major, las_data.header.version.minor)
    
    # 保存原始格式信息
    original_info = {
        "version": current_version,
        "point_format": las_data.header.point_format.id,
        "offsets": las_data.header.offsets.copy(),
        "scales": las_data.header.scales.copy(),
        "vlrs": list(las_data.header.vlrs) if hasattr(las_data.header, 'vlrs') else [],
        "extra_dims": [(dim.name, dim.dtype, getattr(dim, 'description', '')) 
                       for dim in las_data.point_format.extra_dimensions],
        "needs_restore": current_version > target_version
    }
    
    # 如果当前版本高于目标版本，需要转换
    if current_version > target_version:
        from laspy.header import Version
        
        # 创建新的 header，使用兼容的点格式
        # LAS 1.2 支持点格式 0-3，LAS 1.4 支持 0-10
        # LAS 1.4 格式说明：
        # 格式 6: 基本 + GPS Time (等价于格式 1)
        # 格式 7: 基本 + GPS Time + RGB (等价于格式 3)
        # 格式 8: 基本 + GPS Time + RGB + NIR (等价于格式 3 + NIR)
        current_point_format = las_data.header.point_format.id
        
        # 如果点格式 > 3，需要降级到格式 3（带 GPS 时间和 RGB）或更低
        if current_point_format > 3:
            # 检查是否有 RGB 数据和 GPS 时间
            has_rgb = hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue')
            has_gps = hasattr(las_data, 'gps_time')
            
            # LAS 1.4 格式 6+ 都包含 GPS Time，优先保留
            if has_rgb and has_gps:
                new_point_format = 3  # GPS Time + RGB
            elif has_rgb:
                new_point_format = 3  # 即使没检测到 GPS，格式 6+ 应该都有，选择格式 3
            elif has_gps:
                new_point_format = 1  # 只有 GPS Time
            else:
                # 格式 6+ 理论上都有 GPS Time，保险起见选择格式 1
                new_point_format = 1
        else:
            new_point_format = current_point_format
        
        # 创建新 header
        new_header = laspy.LasHeader(point_format=new_point_format, version=Version(*target_version))
        new_header.offsets = las_data.header.offsets
        new_header.scales = las_data.header.scales
        
        # 复制 VLRs（只复制兼容的）
        if hasattr(las_data.header, 'vlrs'):
            for vlr in las_data.header.vlrs:
                try:
                    new_header.vlrs.append(vlr)
                except:
                    pass  # 跳过不兼容的 VLR
        
        # 添加 extra dimensions 来保存 LAS 1.4 特有的属性
        # 注意：LAS extra bytes name 最大 32 字符，description 最大 32 字符
        # 保存 scan_angle (LAS 1.4 使用 scan_angle, LAS 1.2 使用 scan_angle_rank)
        if current_point_format >= 6 and hasattr(las_data, 'scan_angle'):
            new_header.add_extra_dim(laspy.ExtraBytesParams(
                name="_scan_ang",
                type=np.float32,
                description=""
            ))
        
        # 保存 scanner_channel (LAS 1.4 特有)
        if hasattr(las_data, 'scanner_channel'):
            new_header.add_extra_dim(laspy.ExtraBytesParams(
                name="_scan_ch",
                type=np.uint8,
                description=""
            ))
        
        # 保存 NIR (近红外, LAS 1.4 点格式 8, 10)
        if hasattr(las_data, 'nir'):
            new_header.add_extra_dim(laspy.ExtraBytesParams(
                name="_nir",
                type=np.uint16,
                description=""
            ))
        
        # 复制原始的 extra dimensions
        for dim in las_data.point_format.extra_dimensions:
            try:
                # 截断名称和描述以符合 LAS 规范 (最大 32 字符)
                dim_name = dim.name[:32] if len(dim.name) > 32 else dim.name
                dim_desc = getattr(dim, 'description', '')
                dim_desc = dim_desc[:32] if len(dim_desc) > 32 else dim_desc
                new_header.add_extra_dim(laspy.ExtraBytesParams(
                    name=dim_name,
                    type=dim.dtype,
                    description=dim_desc
                ))
            except:
                pass
        
        # 创建新的 LAS 数据
        new_las = laspy.LasData(new_header)
        new_las.points = laspy.ScaleAwarePointRecord.zeros(len(las_data.points), header=new_header)
        
        # 复制坐标
        new_las.x = las_data.x
        new_las.y = las_data.y
        new_las.z = las_data.z
        
        # 复制基本维度
        basic_dims = ['intensity', 'return_number', 'number_of_returns',
                      'scan_direction_flag', 'edge_of_flight_line', 'classification',
                      'synthetic', 'key_point', 'withheld', 'user_data', 'point_source_id']
        
        for dim in basic_dims:
            if hasattr(las_data, dim) and hasattr(new_las, dim):
                try:
                    setattr(new_las, dim, getattr(las_data, dim))
                except Exception as e:
                    pass
        
        # 处理 scan_angle：LAS 1.4 格式 6+ 使用 scan_angle (float)，LAS 1.2 使用 scan_angle_rank (int8)
        if current_point_format >= 6 and hasattr(las_data, 'scan_angle'):
            # 保存原始 scan_angle 到 extra dim
            if hasattr(new_las, '_scan_ang'):
                new_las._scan_ang = np.array(las_data.scan_angle, dtype=np.float32)
            # 转换为 scan_angle_rank (截断到 -90 到 90 范围)
            if hasattr(new_las, 'scan_angle_rank'):
                scan_angles = np.array(las_data.scan_angle)
                new_las.scan_angle_rank = np.clip(scan_angles, -90, 90).astype(np.int8)
        elif hasattr(las_data, 'scan_angle_rank') and hasattr(new_las, 'scan_angle_rank'):
            new_las.scan_angle_rank = las_data.scan_angle_rank
        
        # 保存 scanner_channel
        if hasattr(las_data, 'scanner_channel') and hasattr(new_las, '_scan_ch'):
            new_las._scan_ch = np.array(las_data.scanner_channel, dtype=np.uint8)
        
        # 保存 NIR
        if hasattr(las_data, 'nir') and hasattr(new_las, '_nir'):
            new_las._nir = np.array(las_data.nir, dtype=np.uint16)
        
        # 复制 GPS 时间（点格式 1, 3, 4, 5, 6, 7, 8, 9, 10 都支持 GPS Time）
        if hasattr(las_data, 'gps_time') and hasattr(new_las, 'gps_time'):
            try:
                new_las.gps_time = las_data.gps_time
            except Exception as e:
                print(f"警告: 无法复制 GPS Time: {e}")
        
        # 复制 RGB（点格式 2, 3, 5, 7, 8, 10 支持 RGB）
        if hasattr(las_data, 'red') and hasattr(new_las, 'red'):
            try:
                new_las.red = las_data.red
                new_las.green = las_data.green
                new_las.blue = las_data.blue
            except Exception as e:
                print(f"警告: 无法复制 RGB: {e}")
        
        # 复制原始的 extra dimensions（使用可能被截断的名称）
        for dim in las_data.point_format.extra_dimensions:
            dim_name = dim.name[:32] if len(dim.name) > 32 else dim.name
            if hasattr(las_data, dim.name) and hasattr(new_las, dim_name):
                try:
                    setattr(new_las, dim_name, getattr(las_data, dim.name))
                except:
                    pass
        
        new_las.update_header()
        new_las.write(output_path)
        
        del las_data, new_las
        return original_info
    else:
        # 无需转换，直接复制
        del las_data
        shutil.copy2(input_path, output_path)
        original_info["needs_restore"] = False
        return original_info


def restore_las_to_original_version(input_path: str, output_path: str, original_info: Dict[str, Any]) -> bool:
    """
    将 DLL 处理后的 LAS 文件还原到原始版本格式
    
    Args:
        input_path: DLL 处理后的 LAS 文件路径
        output_path: 输出 LAS 文件路径
        original_info: 原始格式信息字典
    
    Returns:
        bool: 是否成功还原
    """
    if not original_info.get("needs_restore", False):
        # 无需还原，直接复制
        shutil.copy2(input_path, output_path)
        return True
    
    try:
        from laspy.header import Version
        
        las_data = laspy.read(input_path)
        
        # 创建原始版本的 header
        orig_version = original_info["version"]
        orig_point_format = original_info["point_format"]
        
        new_header = laspy.LasHeader(point_format=orig_point_format, version=Version(*orig_version))
        new_header.offsets = original_info["offsets"]
        new_header.scales = original_info["scales"]
        
        # 复制 VLRs
        for vlr in original_info.get("vlrs", []):
            try:
                new_header.vlrs.append(vlr)
            except:
                pass
        
        # 添加原始的 extra dimensions（排除我们添加的临时属性）
        for dim_name, dim_dtype, dim_desc in original_info.get("extra_dims", []):
            if not dim_name.startswith("_"):  # 排除 _scan_ang, _scan_ch, _nir 等临时属性
                try:
                    # 截断以符合 LAS 规范
                    dim_name_safe = dim_name[:32] if len(dim_name) > 32 else dim_name
                    dim_desc_safe = dim_desc[:32] if len(dim_desc) > 32 else dim_desc
                    new_header.add_extra_dim(laspy.ExtraBytesParams(
                        name=dim_name_safe,
                        type=dim_dtype,
                        description=dim_desc_safe
                    ))
                except:
                    pass
        
        # 创建新的 LAS 数据
        new_las = laspy.LasData(new_header)
        new_las.points = laspy.ScaleAwarePointRecord.zeros(len(las_data.points), header=new_header)
        
        # 复制坐标
        new_las.x = las_data.x
        new_las.y = las_data.y
        new_las.z = las_data.z
        
        # 复制基本维度
        basic_dims = ['intensity', 'return_number', 'number_of_returns',
                      'scan_direction_flag', 'edge_of_flight_line', 'classification',
                      'synthetic', 'key_point', 'withheld', 'user_data', 'point_source_id']
        
        for dim in basic_dims:
            if hasattr(las_data, dim) and hasattr(new_las, dim):
                try:
                    setattr(new_las, dim, getattr(las_data, dim))
                except:
                    pass
        
        # 还原 scan_angle：从保存的 extra dim 或从 scan_angle_rank 转换
        if orig_point_format >= 6 and hasattr(new_las, 'scan_angle'):
            if hasattr(las_data, '_scan_ang'):
                # 使用保存的原始值
                new_las.scan_angle = las_data._scan_ang
            elif hasattr(las_data, 'scan_angle_rank'):
                # 从 scan_angle_rank 转换
                new_las.scan_angle = np.array(las_data.scan_angle_rank, dtype=np.float32)
        elif hasattr(las_data, 'scan_angle_rank') and hasattr(new_las, 'scan_angle_rank'):
            new_las.scan_angle_rank = las_data.scan_angle_rank
        
        # 还原 scanner_channel
        if hasattr(new_las, 'scanner_channel'):
            if hasattr(las_data, '_scan_ch'):
                new_las.scanner_channel = las_data._scan_ch
            else:
                new_las.scanner_channel = np.zeros(len(las_data.points), dtype=np.uint8)
        
        # 还原 NIR
        if hasattr(new_las, 'nir'):
            if hasattr(las_data, '_nir'):
                new_las.nir = las_data._nir
            else:
                new_las.nir = np.zeros(len(las_data.points), dtype=np.uint16)
        
        # 复制 GPS 时间（LAS 1.4 格式 6+ 和 LAS 1.2 格式 1, 3 都支持）
        if hasattr(las_data, 'gps_time') and hasattr(new_las, 'gps_time'):
            try:
                new_las.gps_time = las_data.gps_time
            except Exception as e:
                print(f"警告: 还原时无法复制 GPS Time: {e}")
        
        # 复制 RGB
        if hasattr(las_data, 'red') and hasattr(new_las, 'red'):
            try:
                new_las.red = las_data.red
                new_las.green = las_data.green
                new_las.blue = las_data.blue
            except Exception as e:
                print(f"警告: 还原时无法复制 RGB: {e}")
        
        # 复制 extra dimensions（排除临时属性）
        for dim in las_data.point_format.extra_dimensions:
            if not dim.name.startswith("_") and hasattr(new_las, dim.name):
                try:
                    setattr(new_las, dim.name, getattr(las_data, dim.name))
                except:
                    pass
        
        # 复制原始的 extra dimensions（如果在原始文件中存在）
        for dim_name, dim_dtype, dim_desc in original_info.get("extra_dims", []):
            dim_name_safe = dim_name[:32] if len(dim_name) > 32 else dim_name
            if hasattr(las_data, dim_name_safe) and hasattr(new_las, dim_name_safe):
                try:
                    setattr(new_las, dim_name_safe, getattr(las_data, dim_name_safe))
                except:
                    pass
        
        new_las.update_header()
        new_las.write(output_path)
        
        del las_data, new_las
        return True
        
    except Exception as e:
        print(f"还原 LAS 版本时出错: {e}")
        # 出错时直接复制
        shutil.copy2(input_path, output_path)
        return False


def restore_attributes_from_original(dll_output_path: str, original_input_path: str, output_path: str) -> bool:
    """
    从原始输入文件中恢复 DLL 处理后丢失的属性（GPS Time, RGB 等）
    DLL 只修改 classification，其他属性应该从原始文件恢复
    
    Args:
        dll_output_path: DLL 处理后的输出文件路径
        original_input_path: 原始输入文件路径（DLL 处理前）
        output_path: 最终输出文件路径
    
    Returns:
        bool: 是否成功恢复
    """
    try:
        # 读取 DLL 输出和原始文件
        dll_las = laspy.read(dll_output_path)
        orig_las = laspy.read(original_input_path)
        
        # 检查点数是否一致
        if len(dll_las.points) != len(orig_las.points):
            print(f"警告: DLL 输出点数 ({len(dll_las.points)}) 与原始文件 ({len(orig_las.points)}) 不一致，无法恢复属性")
            shutil.copy2(dll_output_path, output_path)
            return False
        
        # 使用原始文件的 header 和点格式
        version = orig_las.header.version
        if version.major == 1 and version.minor < 2:
            from laspy.header import Version
            version = Version(1, 2)
        
        new_header = laspy.LasHeader(point_format=orig_las.header.point_format.id, version=version)
        new_header.offsets = orig_las.header.offsets
        new_header.scales = orig_las.header.scales
        
        # 复制 extra dimensions
        for extra_dim in orig_las.point_format.extra_dimensions:
            try:
                dim_name = extra_dim.name[:32] if len(extra_dim.name) > 32 else extra_dim.name
                dim_desc = (extra_dim.description if hasattr(extra_dim, 'description') else "")
                dim_desc = dim_desc[:32] if len(dim_desc) > 32 else dim_desc
                new_header.add_extra_dim(laspy.ExtraBytesParams(
                    name=dim_name,
                    type=extra_dim.dtype,
                    description=dim_desc
                ))
            except:
                pass
        
        # 复制 VLRs
        if hasattr(orig_las.header, 'vlrs'):
            for vlr in orig_las.header.vlrs:
                try:
                    new_header.vlrs.append(vlr)
                except:
                    pass
        
        # 创建新的 LAS 数据
        new_las = laspy.LasData(new_header)
        new_las.points = laspy.ScaleAwarePointRecord.zeros(len(orig_las.points), header=new_header)
        
        # 从原始文件复制所有属性
        for dim in orig_las.point_format.dimension_names:
            if hasattr(orig_las, dim) and hasattr(new_las, dim):
                try:
                    setattr(new_las, dim, getattr(orig_las, dim))
                except:
                    pass
        
        # 从原始文件复制 extra dimensions
        for extra_dim in orig_las.point_format.extra_dimensions:
            dim_name = extra_dim.name[:32] if len(extra_dim.name) > 32 else extra_dim.name
            if hasattr(orig_las, dim_name) and hasattr(new_las, dim_name):
                try:
                    setattr(new_las, dim_name, getattr(orig_las, dim_name))
                except:
                    pass
        
        # 从 DLL 输出复制 classification（这是 DLL 唯一修改的内容）
        if hasattr(dll_las, 'classification') and hasattr(new_las, 'classification'):
            new_las.classification = dll_las.classification
        
        new_las.update_header()
        new_las.write(output_path)
        
        del dll_las, orig_las, new_las
        return True
        
    except Exception as e:
        print(f"恢复属性时出错: {e}")
        import traceback
        traceback.print_exc()
        shutil.copy2(dll_output_path, output_path)
        return False


def restore_las_to_original_version_old(input_path: str, output_path: str, original_info: Dict[str, Any]) -> bool:
    """
    将 DLL 处理后的 LAS 文件还原到原始版本格式
    
    Args:
        input_path: DLL 处理后的 LAS 文件路径
        output_path: 输出 LAS 文件路径
        original_info: 原始格式信息字典
    
    Returns:
        bool: 是否成功还原
    """
    if not original_info.get("needs_restore", False):
        # 无需还原，直接复制
        shutil.copy2(input_path, output_path)
        return True
    
    try:
        from laspy.header import Version
        
        las_data = laspy.read(input_path)
        
        # 创建原始版本的 header
        orig_version = original_info["version"]
        orig_point_format = original_info["point_format"]
        
        new_header = laspy.LasHeader(point_format=orig_point_format, version=Version(*orig_version))
        new_header.offsets = original_info["offsets"]
        new_header.scales = original_info["scales"]
        
        # 复制 VLRs
        for vlr in original_info.get("vlrs", []):
            try:
                new_header.vlrs.append(vlr)
            except:
                pass
        
        # 添加原始的 extra dimensions（排除我们添加的临时属性）
        for dim_name, dim_dtype, dim_desc in original_info.get("extra_dims", []):
            if not dim_name.startswith("_"):  # 排除 _scan_ang, _scan_ch, _nir 等临时属性
                try:
                    # 截断以符合 LAS 规范
                    dim_name_safe = dim_name[:32] if len(dim_name) > 32 else dim_name
                    dim_desc_safe = dim_desc[:32] if len(dim_desc) > 32 else dim_desc
                    new_header.add_extra_dim(laspy.ExtraBytesParams(
                        name=dim_name_safe,
                        type=dim_dtype,
                        description=dim_desc_safe
                    ))
                except:
                    pass
        
        # 创建新的 LAS 数据
        new_las = laspy.LasData(new_header)
        new_las.points = laspy.ScaleAwarePointRecord.zeros(len(las_data.points), header=new_header)
        
        # 复制坐标
        new_las.x = las_data.x
        new_las.y = las_data.y
        new_las.z = las_data.z
        
        # 复制基本维度
        basic_dims = ['intensity', 'return_number', 'number_of_returns',
                      'scan_direction_flag', 'edge_of_flight_line', 'classification',
                      'synthetic', 'key_point', 'withheld', 'user_data', 'point_source_id']
        
        for dim in basic_dims:
            if hasattr(las_data, dim) and hasattr(new_las, dim):
                try:
                    setattr(new_las, dim, getattr(las_data, dim))
                except:
                    pass
        
        # 还原 scan_angle：从保存的 extra dim 或从 scan_angle_rank 转换
        if orig_point_format >= 6 and hasattr(new_las, 'scan_angle'):
            if hasattr(las_data, '_scan_ang'):
                # 使用保存的原始值
                new_las.scan_angle = las_data._scan_ang
            elif hasattr(las_data, 'scan_angle_rank'):
                # 从 scan_angle_rank 转换
                new_las.scan_angle = np.array(las_data.scan_angle_rank, dtype=np.float32)
        elif hasattr(las_data, 'scan_angle_rank') and hasattr(new_las, 'scan_angle_rank'):
            new_las.scan_angle_rank = las_data.scan_angle_rank
        
        # 还原 scanner_channel
        if hasattr(new_las, 'scanner_channel'):
            if hasattr(las_data, '_scan_ch'):
                new_las.scanner_channel = las_data._scan_ch
            else:
                new_las.scanner_channel = np.zeros(len(las_data.points), dtype=np.uint8)
        
        # 还原 NIR
        if hasattr(new_las, 'nir'):
            if hasattr(las_data, '_nir'):
                new_las.nir = las_data._nir
            else:
                new_las.nir = np.zeros(len(las_data.points), dtype=np.uint16)
        
        # 复制 GPS 时间（LAS 1.4 格式 6+ 和 LAS 1.2 格式 1, 3 都支持）
        if hasattr(las_data, 'gps_time') and hasattr(new_las, 'gps_time'):
            try:
                new_las.gps_time = las_data.gps_time
            except Exception as e:
                print(f"警告: 还原时无法复制 GPS Time: {e}")
        
        # 复制 RGB
        if hasattr(las_data, 'red') and hasattr(new_las, 'red'):
            try:
                new_las.red = las_data.red
                new_las.green = las_data.green
                new_las.blue = las_data.blue
            except Exception as e:
                print(f"警告: 还原时无法复制 RGB: {e}")
        
        # 复制 extra dimensions（排除临时属性）
        for dim in las_data.point_format.extra_dimensions:
            if not dim.name.startswith("_") and hasattr(new_las, dim.name):
                try:
                    setattr(new_las, dim.name, getattr(las_data, dim.name))
                except:
                    pass
        
        # 复制原始的 extra dimensions（如果在原始文件中存在）
        for dim_name, dim_dtype, dim_desc in original_info.get("extra_dims", []):
            dim_name_safe = dim_name[:32] if len(dim_name) > 32 else dim_name
            if hasattr(las_data, dim_name_safe) and hasattr(new_las, dim_name_safe):
                try:
                    setattr(new_las, dim_name_safe, getattr(las_data, dim_name_safe))
                except:
                    pass
        
        new_las.update_header()
        new_las.write(output_path)
        
        del las_data, new_las
        return True
        
    except Exception as e:
        print(f"还原 LAS 版本时出错: {e}")
        # 出错时直接复制
        shutil.copy2(input_path, output_path)
        return False


def _batch_dll_process(
    process_name: str,
    input_dir: str,
    output_dir: str,
    use_tile: bool = False,
    window_size: Tuple[float, float] = (1000.0, 1000.0),
    min_points: Optional[int] = 10000,
    max_points: Optional[int] = None,
    label_remap: bool = False,
    label_count: bool = False,
    save_sample_weight: bool = False,
    require_labels: Optional[List[int]] = None,
    use_trash_bin: bool = False,
    trash_bin_label: int = 0,
    dll_path: str = DLL_PATH,
    use_den: bool = False,
    den_dll_path: str = DEN_DLL_PATH
) -> dict:
    """
    批量处理 LAS 文件进行 DLL 处理
    
    Args:
        input_dir: 输入 LAS 文件夹路径
        output_dir: 输出文件夹路径
        use_tile: 是否对 LAS 文件进行 tile 分块处理（处理大文件时启用）
        window_size: tile 窗口大小，默认 (1000, 1000)
        min_points: tile 最小点数阈值，默认 10000
        max_points: tile 最大点数阈值，默认 None
        label_remap: 是否重映射标签，默认 False
        label_count: 是否统计标签，默认 False
        save_sample_weight: 是否保存样本权重，默认 False
        require_labels: 需要保留的标签列表，默认 None
        use_trash_bin: 是否启用垃圾桶机制，默认 False
        trash_bin_label: 垃圾桶标签值，默认 0
        dll_path: DLL 文件路径，默认使用配置的路径
    
    Returns:
        处理结果统计字典，包含 success_count, fail_count, failed_files, elapsed_time
    """
    process_name = process_name.upper()
    process_name_lower = process_name.lower()
    result = {
        "success_count": 0,
        "fail_count": 0,
        "failed_files": [],
        "elapsed_time": 0.0
    }
    
    # 检查 DLL
    if not os.path.exists(dll_path):
        print(f"错误：找不到 DLL 文件 -> {dll_path}")
        return result

    # 加载 DLL
    try:
        lib = ctypes.CDLL(dll_path)
        lib.StartProcess.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        print(">>> DLL 加载成功")
    except Exception as e:
        print(f"DLL 加载失败: {e}")
        return result

    # 准备输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f">>> 创建输出目录: {output_dir}")

    start_time = time.time()

    if use_tile:
        # 使用 tile 模式处理大文件
        print(">>> 启用 tile 分块处理模式")
        
        from utils.las_tile import process_las_tiles
        from utils.las_merge import merge_las_segments
        
        # 创建临时目录存放中间文件
        temp_tile_dir = tempfile.mkdtemp(prefix=f"{process_name_lower}_tile_")
        temp_lac_dir = tempfile.mkdtemp(prefix=f"{process_name_lower}_result_")
        temp_den_dir = None
        
        try:
            # Step 1: 对输入文件进行 tile 分块
            print(f"\n>>> Step 1: Tile 分块处理...")
            process_las_tiles(
                input_path=input_dir,
                output_dir=temp_tile_dir,
                window_size=window_size,
                min_points=min_points,
                max_points=max_points,
                label_remap=label_remap,
                label_count=label_count,
                save_sample_weight=save_sample_weight,
                require_labels=require_labels,
                use_trash_bin=use_trash_bin,
                trash_bin_label=trash_bin_label
            )
            
            # Step 2: 对 tile 后的文件进行处理
            print(f"\n>>> Step 2: {process_name} 处理...")
            las_files = [f for f in os.listdir(temp_tile_dir) if f.lower().endswith('.las')]
            total_files = len(las_files)
            
            if total_files == 0:
                print(">>> 未找到 tile 后的 LAS 文件。")
                return result
            
            print(f">>> 发现 {total_files} 个 tile 文件")
            
            for filename in tqdm(las_files, desc=f"{process_name} 处理", unit="file"):
                in_path = os.path.join(temp_tile_dir, filename)
                out_path = os.path.join(temp_lac_dir, filename)
                
                try:
                    # 检查并转换 LAS 版本以兼容 DLL，同时保存原始格式信息
                    temp_converted = None
                    las_check = laspy.read(in_path)
                    current_version = (las_check.header.version.major, las_check.header.version.minor)
                    del las_check
                    
                    original_info = None
                    if current_version > DLL_MAX_LAS_VERSION:
                        # 需要转换版本
                        temp_converted = os.path.join(temp_tile_dir, f"_converted_{filename}")
                        original_info = convert_las_to_compatible_version(in_path, temp_converted, DLL_MAX_LAS_VERSION)
                        actual_in_path = temp_converted
                    else:
                        actual_in_path = in_path
                    
                    # 调用 DLL 处理
                    temp_dll_out = os.path.join(temp_lac_dir, f"_dll_{filename}")
                    b_in = actual_in_path.encode('gbk')
                    b_out = temp_dll_out.encode('gbk')
                    lib.StartProcess(b_in, b_out)
                    
                    # 从原始输入文件恢复所有属性（GPS Time, RGB等），只保留DLL修改的classification
                    temp_restored = os.path.join(temp_lac_dir, f"_restored_{filename}")
                    restore_attributes_from_original(temp_dll_out, in_path, temp_restored)
                    os.remove(temp_dll_out)
                    
                    # 如果需要还原版本（针对1.4格式）
                    if original_info and original_info.get("needs_restore", False):
                        restore_las_to_original_version_old(temp_restored, out_path, original_info)
                        os.remove(temp_restored)
                    else:
                        shutil.move(temp_restored, out_path)
                    
                    result["success_count"] += 1
                    
                    # 清理临时转换文件
                    if temp_converted and os.path.exists(temp_converted):
                        os.remove(temp_converted)
                except Exception as e:
                    result["fail_count"] += 1
                    result["failed_files"].append((filename, str(e)))
                    # 清理临时文件
                    if temp_converted and os.path.exists(temp_converted):
                        try:
                            os.remove(temp_converted)
                        except:
                            pass
                    if os.path.exists(os.path.join(temp_lac_dir, f"_dll_{filename}")):
                        try:
                            os.remove(os.path.join(temp_lac_dir, f"_dll_{filename}"))
                        except:
                            pass
                    if os.path.exists(os.path.join(temp_lac_dir, f"_restored_{filename}")):
                        try:
                            os.remove(os.path.join(temp_lac_dir, f"_restored_{filename}"))
                        except:
                            pass

            merge_input_dir = temp_lac_dir
            if use_den:
                temp_den_dir = tempfile.mkdtemp(prefix=f"{process_name_lower}_den_")
                print(f"\n>>> Step 3: DEN 处理...")
                den_files = [f for f in os.listdir(temp_lac_dir) if f.lower().endswith('.las')]
                den_total = len(den_files)
                if den_total == 0:
                    print(">>> 未找到 LAC 后的 LAS 文件。")
                else:
                    print(f">>> 发现 {den_total} 个 LAC 文件")
                    for filename in tqdm(den_files, desc="DEN 处理", unit="file"):
                        in_path = os.path.join(temp_lac_dir, filename)
                        out_path = os.path.join(temp_den_dir, filename)
                        try:
                            temp_converted = None
                            las_check = laspy.read(in_path)
                            current_version = (las_check.header.version.major, las_check.header.version.minor)
                            del las_check

                            original_info = None
                            if current_version > DLL_MAX_LAS_VERSION:
                                temp_converted = os.path.join(temp_lac_dir, f"_converted_den_{filename}")
                                original_info = convert_las_to_compatible_version(in_path, temp_converted, DLL_MAX_LAS_VERSION)
                                actual_in_path = temp_converted
                            else:
                                actual_in_path = in_path

                            temp_dll_out = os.path.join(temp_den_dir, f"_dll_{filename}")
                            b_in = actual_in_path.encode('gbk')
                            b_out = temp_dll_out.encode('gbk')
                            den_lib = ctypes.CDLL(den_dll_path)
                            den_lib.StartProcess.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                            den_lib.StartProcess(b_in, b_out)

                            temp_restored = os.path.join(temp_den_dir, f"_restored_{filename}")
                            restore_attributes_from_original(temp_dll_out, in_path, temp_restored)
                            os.remove(temp_dll_out)

                            if original_info and original_info.get("needs_restore", False):
                                restore_las_to_original_version_old(temp_restored, out_path, original_info)
                                os.remove(temp_restored)
                            else:
                                shutil.move(temp_restored, out_path)

                            if temp_converted and os.path.exists(temp_converted):
                                os.remove(temp_converted)
                        except Exception as e:
                            result["fail_count"] += 1
                            result["failed_files"].append((filename, str(e)))
                            if temp_converted and os.path.exists(temp_converted):
                                try:
                                    os.remove(temp_converted)
                                except:
                                    pass
                            if os.path.exists(os.path.join(temp_den_dir, f"_dll_{filename}")):
                                try:
                                    os.remove(os.path.join(temp_den_dir, f"_dll_{filename}"))
                                except:
                                    pass
                            if os.path.exists(os.path.join(temp_den_dir, f"_restored_{filename}")):
                                try:
                                    os.remove(os.path.join(temp_den_dir, f"_restored_{filename}"))
                                except:
                                    pass
                merge_input_dir = temp_den_dir if den_total > 0 else temp_lac_dir

            # Step 4: 合并处理后的文件
            print(f"\n>>> Step 4: 合并结果...")
            merge_las_segments(
                input_path=merge_input_dir,
                output_dir=output_dir,
                label_remap_file=None
            )
            
        finally:
            # 清理临时目录
            print(f"\n>>> 清理临时文件...")
            if os.path.exists(temp_tile_dir):
                shutil.rmtree(temp_tile_dir)
            if os.path.exists(temp_lac_dir):
                shutil.rmtree(temp_lac_dir)
            if temp_den_dir and os.path.exists(temp_den_dir):
                shutil.rmtree(temp_den_dir)
    else:
        # 直接处理模式
        las_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.las')]
        total_files = len(las_files)

        if total_files == 0:
            print(">>> 未找到 LAS 文件。")
            return result

        print(f">>> 发现 {total_files} 个文件")

        for filename in tqdm(las_files, desc=f"{process_name} 处理", unit="file"):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename)
            
            try:
                # 检查并转换 LAS 版本以兼容 DLL，同时保存原始格式信息
                temp_converted = None
                las_check = laspy.read(in_path)
                current_version = (las_check.header.version.major, las_check.header.version.minor)
                del las_check
                
                original_info = None
                if current_version > DLL_MAX_LAS_VERSION:
                    # 需要转换版本
                    temp_converted = os.path.join(input_dir, f"_converted_{filename}")
                    original_info = convert_las_to_compatible_version(in_path, temp_converted, DLL_MAX_LAS_VERSION)
                    actual_in_path = temp_converted
                else:
                    actual_in_path = in_path
                
                # 调用 DLL 处理
                temp_dll_out = os.path.join(output_dir, f"_dll_{filename}")
                b_in = actual_in_path.encode('gbk')
                b_out = temp_dll_out.encode('gbk')
                lib.StartProcess(b_in, b_out)
                
                # 从原始输入文件恢复所有属性（GPS Time, RGB等），只保留DLL修改的classification
                temp_restored = os.path.join(output_dir, f"_restored_{filename}")
                restore_attributes_from_original(temp_dll_out, in_path, temp_restored)
                os.remove(temp_dll_out)
                
                # 如果需要还原版本（针对1.4格式）
                if original_info and original_info.get("needs_restore", False):
                    restore_las_to_original_version_old(temp_restored, out_path, original_info)
                    os.remove(temp_restored)
                else:
                    shutil.move(temp_restored, out_path)
                
                result["success_count"] += 1
                
                # 清理临时转换文件
                if temp_converted and os.path.exists(temp_converted):
                    os.remove(temp_converted)
            except Exception as e:
                result["fail_count"] += 1
                result["failed_files"].append((filename, str(e)))
                # 清理临时文件
                if temp_converted and os.path.exists(temp_converted):
                    try:
                        os.remove(temp_converted)
                    except:
                        pass
                if os.path.exists(os.path.join(output_dir, f"_dll_{filename}")):
                    try:
                        os.remove(os.path.join(output_dir, f"_dll_{filename}"))
                    except:
                        pass
                if os.path.exists(os.path.join(output_dir, f"_restored_{filename}")):
                    try:
                        os.remove(os.path.join(output_dir, f"_restored_{filename}"))
                    except:
                        pass
                if temp_converted and os.path.exists(temp_converted):
                    try:
                        os.remove(temp_converted)
                    except:
                        pass
                if os.path.exists(os.path.join(output_dir, f"_dll_{filename}")):
                    try:
                        os.remove(os.path.join(output_dir, f"_dll_{filename}"))
                    except:
                        pass

        if use_den:
            den_output_dir = tempfile.mkdtemp(prefix=f"{process_name_lower}_den_")
            try:
                print(f"\n>>> Step 3: DEN 处理...")
                den_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.las')]
                den_total = len(den_files)
                if den_total == 0:
                    print(">>> 未找到 LAC 后的 LAS 文件。")
                else:
                    print(f">>> 发现 {den_total} 个 LAC 文件")
                    den_lib = ctypes.CDLL(den_dll_path)
                    den_lib.StartProcess.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                    for filename in tqdm(den_files, desc="DEN 处理", unit="file"):
                        in_path = os.path.join(output_dir, filename)
                        out_path = os.path.join(den_output_dir, filename)
                        try:
                            temp_converted = None
                            las_check = laspy.read(in_path)
                            current_version = (las_check.header.version.major, las_check.header.version.minor)
                            del las_check

                            original_info = None
                            if current_version > DLL_MAX_LAS_VERSION:
                                temp_converted = os.path.join(output_dir, f"_converted_den_{filename}")
                                original_info = convert_las_to_compatible_version(in_path, temp_converted, DLL_MAX_LAS_VERSION)
                                actual_in_path = temp_converted
                            else:
                                actual_in_path = in_path

                            temp_dll_out = os.path.join(den_output_dir, f"_dll_{filename}")
                            b_in = actual_in_path.encode('gbk')
                            b_out = temp_dll_out.encode('gbk')
                            den_lib.StartProcess(b_in, b_out)

                            temp_restored = os.path.join(den_output_dir, f"_restored_{filename}")
                            restore_attributes_from_original(temp_dll_out, in_path, temp_restored)
                            os.remove(temp_dll_out)

                            if original_info and original_info.get("needs_restore", False):
                                restore_las_to_original_version_old(temp_restored, out_path, original_info)
                                os.remove(temp_restored)
                            else:
                                shutil.move(temp_restored, out_path)

                            if temp_converted and os.path.exists(temp_converted):
                                os.remove(temp_converted)
                        except Exception as e:
                            result["fail_count"] += 1
                            result["failed_files"].append((filename, str(e)))
                for filename in os.listdir(den_output_dir):
                    if filename.lower().endswith('.las'):
                        shutil.copy2(os.path.join(den_output_dir, filename), os.path.join(output_dir, filename))
            finally:
                if os.path.exists(den_output_dir):
                    shutil.rmtree(den_output_dir)

    # 结束
    result["elapsed_time"] = time.time() - start_time
    
    print(f"\n>>> 全部完成！耗时: {result['elapsed_time']:.2f} 秒")
    print(f">>> 成功: {result['success_count']}, 失败: {result['fail_count']}")
    print(f">>> 结果已保存在: {output_dir}")
    
    if result["failed_files"]:
        print("\n>>> 失败文件列表:")
        for fname, err in result["failed_files"]:
            print(f"    - {fname}: {err}")
    
    return result


def batch_lac_process(
    input_dir: str,
    output_dir: str,
    use_tile: bool = False,
    window_size: Tuple[float, float] = (1000.0, 1000.0),
    min_points: Optional[int] = 10000,
    max_points: Optional[int] = None,
    label_remap: bool = False,
    label_count: bool = False,
    save_sample_weight: bool = False,
    require_labels: Optional[List[int]] = None,
    use_trash_bin: bool = False,
    trash_bin_label: int = 0,
    dll_path: str = DLL_PATH,
    use_den: bool = False,
    den_dll_path: str = DEN_DLL_PATH
) -> dict:
    result = _batch_dll_process(
        "LAC",
        input_dir=input_dir,
        output_dir=output_dir,
        use_tile=use_tile,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        label_remap=label_remap,
        label_count=label_count,
        save_sample_weight=save_sample_weight,
        require_labels=require_labels,
        use_trash_bin=use_trash_bin,
        trash_bin_label=trash_bin_label,
        dll_path=dll_path,
        use_den=use_den,
        den_dll_path=den_dll_path,
    )
    return result


def batch_den_process(
    input_dir: str,
    output_dir: str,
    use_tile: bool = False,
    window_size: Tuple[float, float] = (1000.0, 1000.0),
    min_points: Optional[int] = 10000,
    max_points: Optional[int] = None,
    label_remap: bool = False,
    label_count: bool = False,
    save_sample_weight: bool = False,
    require_labels: Optional[List[int]] = None,
    use_trash_bin: bool = False,
    trash_bin_label: int = 0,
    dll_path: str = DEN_DLL_PATH
) -> dict:
    return _batch_dll_process(
        "DEN",
        input_dir=input_dir,
        output_dir=output_dir,
        use_tile=use_tile,
        window_size=window_size,
        min_points=min_points,
        max_points=max_points,
        label_remap=label_remap,
        label_count=label_count,
        save_sample_weight=save_sample_weight,
        require_labels=require_labels,
        use_trash_bin=use_trash_bin,
        trash_bin_label=trash_bin_label,
        dll_path=dll_path,
    )


if __name__ == "__main__":
    # 示例用法
    INPUT_DIR = r"E:\data\梯田\output3\output_filtered\KM35.las"
    OUTPUT_DIR = r"E:\data\梯田\output3\output_filtered\KM35"
    
    batch_lac_process(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        use_tile=False,  # 启用 tile 分块处理
        window_size=(1000.0, 1000.0),  # tile 窗口大小
        min_points=1000,  # 最小点数阈值
    )
