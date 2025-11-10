"""
Training script with OpenMP fix
解决 OpenMP 多重初始化问题
"""
import os
import sys

# 设置环境变量解决 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 可选：减少 OpenMP 线程数以避免冲突
os.environ['OMP_NUM_THREADS'] = '4'

# 导入并运行原始训练脚本
if __name__ == '__main__':
    # 将 tools 目录添加到路径
    tools_path = os.path.join(os.path.dirname(__file__), 'tools')
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    
    # 导入并运行训练脚本
    from tools.train import main
    main()
