"""
CLIP 图像检索系统 - 简单示例

这个示例展示了如何使用CLIP模型进行图像检索的基本用法
"""

import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

def simple_demo():
    """简单演示如何使用CLIP图像检索系统"""
    
    try:
        # 由于文件名包含连字符，需要使用importlib导入
        import importlib.util
        spec = importlib.util.spec_from_file_location("clip_retrieval", "CLIP_Image_Retrieval.py")
        clip_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clip_module)
        CLIPImageRetrieval = clip_module.CLIPImageRetrieval
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please make sure you have installed all required dependencies.")
        print("Run: bash setup_exercise2.sh")
        return
    
    print("CLIP Image Retrieval - Simple Demo")
    print("=" * 40)
    
    # 1. 初始化系统
    retriever = CLIPImageRetrieval()
    
    # 2. 设置图像目录（请根据实际情况修改路径）
    image_dir = "../ResNet-50_Fine-Tuning/data"
    
    if not os.path.exists(image_dir):
        print(f"Warning: Image directory not found: {image_dir}")
        print("Please make sure you have images in the specified directory.")
        return
    
    # 3. 构建图像索引
    print("Building image index...")
    retriever.build_image_index(image_dir)
    
    if len(retriever.image_paths) == 0:
        print("No images found in the database!")
        return
    
    print(f"Successfully indexed {len(retriever.image_paths)} images")
    
    # 4. 选择一张查询图像（使用数据库中的第一张图像作为示例）
    query_image = retriever.image_paths[0]
    print(f"Using query image: {os.path.basename(query_image)}")
    
    # 5. 搜索相似图像
    results = retriever.search_similar_images(query_image, top_k=5)
    
    # 6. 显示结果
    if results:
        print("\nSearch Results:")
        print("-" * 40)
        for result in results:
            print(f"Rank {result['rank']}: {result['filename']} "
                  f"(Similarity: {result['similarity']:.4f})")
        
        # 7. 可视化结果
        print("\nGenerating visualization...")
        retriever.visualize_search_results(query_image, results, 
                                         save_path="./results/demo_result.png")
        print("Visualization saved to: ./results/demo_result.png")
    else:
        print("No search results found!")

def check_dependencies():
    """检查依赖是否正确安装"""
    print("Checking dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'clip': 'OpenAI CLIP',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please run: bash setup_exercise2.sh")
        return False
    else:
        print("\nAll dependencies are installed!")
        return True

if __name__ == "__main__":
    # 检查依赖
    if check_dependencies():
        # 运行演示
        simple_demo()
    else:
        print("Please install missing dependencies first.")
