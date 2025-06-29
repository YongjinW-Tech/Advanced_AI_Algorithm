#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced AI Algorithm 项目环境配置脚本

此脚本用于自动安装项目所需的所有依赖包
支持检测环境并自动安装缺失的包

作者：YjTech
日期：2025年6月29日
"""

import subprocess
import sys
import importlib.util

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """安装指定包"""
    print(f"正在安装 {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package_name} 安装失败")
        return False

def main():
    """主安装流程"""
    print("🚀 Advanced AI Algorithm 项目环境配置")
    print("=" * 50)
    
    # 定义需要检查的包列表 (包名, 导入名)
    packages = [
        ("pandas>=1.5.0", "pandas"),
        ("numpy>=1.21.0", "numpy"),
        ("matplotlib>=3.5.0", "matplotlib"),
        ("seaborn>=0.11.0", "seaborn"),
        ("scikit-learn>=1.0.0", "sklearn"),
        ("sentence-transformers>=2.0.0", "sentence_transformers"),
        ("umap-learn>=0.5.0", "umap"),
        ("tqdm>=4.64.0", "tqdm"),
    ]
    
    print("📦 检查依赖包...")
    
    missing_packages = []
    installed_packages = []
    
    for package_spec, import_name in packages:
        package_name = package_spec.split(">=")[0]
        if check_package(package_name, import_name):
            print(f"✅ {package_name} 已安装")
            installed_packages.append(package_name)
        else:
            print(f"❌ {package_name} 未安装")
            missing_packages.append(package_spec)
    
    if not missing_packages:
        print("\n🎉 所有依赖包都已安装！")
        print("\n您可以运行以下命令来验证安装：")
        print("python -c \"import pandas, numpy, matplotlib, seaborn, sklearn, sentence_transformers, umap, tqdm; print('所有包导入成功!')\"")
        return
    
    print(f"\n⚠️  发现 {len(missing_packages)} 个缺失的包")
    
    # 询问是否安装
    user_input = input("\n是否现在安装缺失的包？(y/n): ").strip().lower()
    
    if user_input in ['y', 'yes', '是']:
        print("\n🔧 开始安装缺失的包...")
        
        success_count = 0
        for package_spec in missing_packages:
            if install_package(package_spec):
                success_count += 1
        
        print(f"\n📊 安装完成！成功安装 {success_count}/{len(missing_packages)} 个包")
        
        if success_count == len(missing_packages):
            print("🎉 所有依赖包安装成功！")
            print("\n💡 建议下载模型（首次运行时会自动下载）：")
            print("运行任一Exercise-1_4*.py文件来下载BGE-m3模型")
        else:
            print("⚠️  部分包安装失败，请手动安装或检查网络连接")
    else:
        print("\n📋 您可以手动安装缺失的包：")
        print("pip install " + " ".join([f'"{pkg}"' for pkg in missing_packages]))
    
    print("\n📚 更多信息请查看 DEPENDENCIES.md 文件")

if __name__ == "__main__":
    main()
