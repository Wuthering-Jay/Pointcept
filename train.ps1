# PowerShell script to run training with OpenMP fix
# 解决 OpenMP 多重初始化问题

# 设置环境变量
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "4"

# 运行训练脚本
& python tools/train.py @args
