import os
import sys
import subprocess
import tempfile
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def setup_vs2019_environment():
    """设置VS 2019编译环境"""
    
    # vcvarsall.bat路径
    vcvars_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    
    if not os.path.exists(vcvars_path):
        print("错误: 找不到vcvarsall.bat")
        print("请确保已安装Visual Studio 2022")
        sys.exit(1)
    
    # 创建临时批处理文件来获取环境变量
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as f:
        bat_file = f.name
        f.write(f'@echo off\n')
        f.write(f'call "{vcvars_path}" x64 -vcvars_ver=14.29\n')
        f.write(f'set\n')  # 输出所有环境变量
    
    try:
        # 运行批处理并捕获环境变量
        result = subprocess.run(
            bat_file,
            capture_output=True,
            text=True,
            shell=True,
            encoding='gbk'  # Windows中文环境编码
        )
        
        # 解析环境变量
        env_vars = {}
        for line in result.stdout.split('\n'):
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value
        
        # 更新当前进程环境变量
        for key, value in env_vars.items():
            if key and value:
                os.environ[key] = value
                print(f"设置: {key}=...")
        
        # 特别设置几个关键变量
        os.environ["VCToolsVersion"] = "14.29.30133"
        os.environ["Platform"] = "x64"
        os.environ["PreferredToolArchitecture"] = "x64"
        
        print(f"\n✓ VS 2019环境设置完成")
        print(f"  PATH前100字符: {os.environ.get('PATH', '')[:100]}...")
        
    finally:
        if os.path.exists(bat_file):
            os.unlink(bat_file)

# 设置环境
print("设置VS 2019编译环境...")
setup_vs2019_environment()

# 验证环境
print("\n验证编译器...")
try:
    result = subprocess.run(["cl.exe"], capture_output=True, text=True, shell=True)
    for line in result.stderr.split('\n'):
        if 'Version' in line:
            print(f"编译器版本: {line.strip()}")
            if '14.29' in line:
                print("✓ 使用的是VS 2019工具集 (v142)")
except Exception as e:
    print(f"错误: 无法验证编译器 - {e}")

# 正常setup代码
src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]

extra_compile_args = {
    "cxx": [
        "/wd4624", "/wd4819", "/wd4996", "/wd4503",
        "/O2", "/EHsc", "/MD",
        "/D_WIN32_WINNT=0x0A00",
        "/DNTDDI_VERSION=NTDDI_WIN10_RS2",
        "/DNOMINMAX",
        "/D_SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING",
    ],
    "nvcc": [
        "-O2",
        "-allow-unsupported-compiler",
        "-Xcompiler", "/wd4624",
        "-Xcompiler", "/wd4819",
        "-Xcompiler", "/wd4996",
        "-Xcompiler", "/EHsc",
        "-Xcompiler", "/MD",
        "-Xcompiler", "/D_WIN32_WINNT=0x0A00",
        "-Xcompiler", "/DNOMINMAX",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_86,code=sm_86",
    ]
}

setup(
    name="pointops",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["pointops"],
    package_dir={"pointops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointops._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)