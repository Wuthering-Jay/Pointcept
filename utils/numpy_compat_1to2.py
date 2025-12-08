"""
让NumPy 1.x能加载NumPy 2.x保存的.pth文件
"""
import sys
import types

# 只在NumPy 1.x环境下才需要这个补丁
try:
    import numpy as np
    np_version = tuple(map(int, np.__version__.split('.')[:2]))
    
    if np_version[0] == 1:
        print(f"NumPy {np.__version__} 检测到，应用NumPy 2.x兼容层")
        
        # 创建一个虚拟的 numpy._core 模块
        class NumpyCoreCompat:
            def __init__(self):
                # 导入实际的numpy.core
                import numpy.core as real_core
                self._real_core = real_core
            
            def __getattr__(self, name):
                # 将numpy._core.xxx 重定向到 numpy.core.xxx
                if hasattr(self._real_core, name):
                    return getattr(self._real_core, name)
                elif name == '__all__':
                    return dir(self._real_core)
                else:
                    # 尝试从numpy本身获取
                    if hasattr(np, name):
                        return getattr(np, name)
                    raise AttributeError(f"module 'numpy._core' has no attribute '{name}'")
        
        # 创建模块
        numpy_core_compat = NumpyCoreCompat()
        numpy_core_module = types.ModuleType('numpy._core')
        
        # 复制所有属性
        for attr in dir(np.core):
            if not attr.startswith('_'):
                try:
                    setattr(numpy_core_module, attr, getattr(np.core, attr))
                except:
                    pass
        
        # 添加到sys.modules
        sys.modules['numpy._core'] = numpy_core_module
        
        # 还需要其他相关模块
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
        
        print("NumPy 2.x兼容层已应用")
        
except Exception as e:
    print(f"应用兼容层失败: {e}")