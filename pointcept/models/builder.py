"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

# 构建 models 和 modules 的注册器实例
MODELS = Registry("models")
MODULES = Registry("modules")

# 返回 cfg 配置的 models 实例
def build_model(cfg):
    """Build models."""
    return MODELS.build(cfg)
