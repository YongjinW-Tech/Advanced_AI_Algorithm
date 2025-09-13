#!/usr/bin/env python
"""
项目验证脚本

这个脚本检查所有模块是否能正确导入，验证项目的完整性。
"""

import sys
import importlib

def test_imports():
    """测试所有模块的导入"""
    
    print("🔍 验证项目模块导入...")
    
    modules_to_test = [
        'data_loader',
        'clip_model', 
        'train',
        'evaluation',
        'visualization'
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"⚠️  {module_name}: {e}")
    
    if failed_imports:
        print(f"\n⚠️  有 {len(failed_imports)} 个模块导入失败:")
        for module in failed_imports:
            print(f"   - {module}")
        print("\n建议:")
        print("1. 检查是否安装了所需依赖: pip install -r requirements.txt")
        print("2. 运行安装脚本: ./setup.sh")
    else:
        print(f"\n🎉 所有 {len(modules_to_test)} 个模块导入成功!")
    
    return len(failed_imports) == 0

def test_dependencies():
    """测试关键依赖包"""
    
    print("\n🔍 验证关键依赖包...")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'), 
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow'),
        ('tqdm', 'TQDM')
    ]
    
    missing_deps = []
    
    for dep_name, display_name in dependencies:
        try:
            importlib.import_module(dep_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name}")
            missing_deps.append(display_name)
    
    # 特殊检查CLIP
    try:
        import clip
        print("✅ OpenAI CLIP")
    except ImportError:
        print("❌ OpenAI CLIP")
        missing_deps.append("OpenAI CLIP")
    
    if missing_deps:
        print(f"\n⚠️  缺少 {len(missing_deps)} 个依赖包:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n安装命令:")
        print("pip install -r requirements.txt")
        print("pip install git+https://github.com/openai/CLIP.git")
    else:
        print(f"\n🎉 所有关键依赖包都已安装!")
    
    return len(missing_deps) == 0

def test_torch_setup():
    """测试PyTorch设置"""
    
    print("\n🔍 验证PyTorch设置...")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
        else:
            print("ℹ️  CUDA不可用，将使用CPU训练")
        
        # 测试基本操作
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print("✅ PyTorch基本操作正常")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch设置有问题: {e}")
        return False

def check_file_structure():
    """检查文件结构"""
    
    print("\n🔍 验证文件结构...")
    
    import os
    
    required_files = [
        'README.md',
        'requirements.txt',
        'config.json',
        'setup.sh',
        'data_loader.py',
        'clip_model.py',
        'train.py',
        'evaluation.py',
        'visualization.py',
        'demo.py'
    ]
    
    required_dirs = [
        'data',
        'model-checkpoint', 
        'results'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing_files.append(file_name)
        else:
            print(f"✅ {file_name}")
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
        else:
            print(f"✅ {dir_name}/")
    
    if missing_files or missing_dirs:
        print(f"\n⚠️  缺少文件或目录:")
        for item in missing_files + missing_dirs:
            print(f"   - {item}")
    else:
        print(f"\n🎉 文件结构完整!")
    
    return len(missing_files) == 0 and len(missing_dirs) == 0

def main():
    """主验证函数"""
    
    print("=" * 60)
    print("🔧 多模态CLIP项目验证")
    print("=" * 60)
    
    # 运行各项检查
    checks = [
        ("文件结构", check_file_structure),
        ("依赖包", test_dependencies),
        ("PyTorch设置", test_torch_setup),
        ("模块导入", test_imports)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name}检查失败: {e}")
            results.append((check_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📋 验证结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name:12} {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("🎉 项目验证完全通过! 可以开始使用了。")
        print("\n🚀 下一步:")
        print("1. 运行演示: python demo.py")
        print("2. 查看文档: cat README.md")
        print("3. 开始训练: python train.py")
    else:
        print("⚠️  项目验证未完全通过，请根据上述提示解决问题。")
        print("\n🔧 建议:")
        print("1. 运行安装脚本: ./setup.sh")
        print("2. 手动安装依赖: pip install -r requirements.txt")
        print("3. 重新运行验证: python verify.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
