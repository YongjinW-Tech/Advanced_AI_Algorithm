"""
练习 2：使用 CLIP 构建图像检索系统（以图搜图）

功能：
1. 使用 CLIP 模型提取图像特征向量
2. 构建图像索引库
3. 实现以图搜图功能
4. 可视化检索结果

作者：YongjinW-Tech
日期：2025年9月5日
"""

import torch
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image
import os
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class CLIPImageRetrieval:
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        初始化CLIP图像检索系统
        
        Args:
            model_name: CLIP模型名称，可选: "ViT-B/32", "ViT-B/16", "ViT-L/14"
            device: 计算设备，默认自动选择
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Loading CLIP model: {model_name}")
        print(f"Using device: {self.device}")
        
        # 加载CLIP模型和预处理器
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval() # 设置为评估模式
        
        # 图像索引库
        self.image_features = np.array([])  # 初始化为空的NumPy数组
        self.image_paths = []
        self.feature_dim = None
        
        print(f"CLIP model loaded successfully!")
    
    def extract_image_features(self, image_path):
        """
        提取单张图像的特征向量
        
        Args:
            image_path: 图像路径
            
        Returns:
            numpy.ndarray: 归一化的特征向量
        """
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # L2归一化 ：目的是将特征向量的长度归一化为1，便于后续计算余弦相似度
                image_features = F.normalize(image_features, p=2, dim=1)
            
            return image_features.cpu().numpy().flatten() # 转为一维数组
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def build_image_index(self, image_directory, supported_formats=('.jpg', '.jpeg', '.png', '.bmp')):
        """
        构建图像索引库
        
        Args:
            image_directory: 图像目录路径
            supported_formats: 支持的图像格式
        """
        print(f"Building image index from: {image_directory}")
        
        # 收集所有图像文件
        image_files = []
        for root, dirs, files in os.walk(image_directory):
            for file in files:
                if file.lower().endswith(supported_formats):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print("No image files found!")
            return
        
        print(f"Found {len(image_files)} images")
        
        # 提取特征
        valid_features = []
        valid_paths = []
        
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            features = self.extract_image_features(image_path)
            if features is not None:
                valid_features.append(features)
                valid_paths.append(image_path)
        
        if valid_features:
            self.image_features = np.array(valid_features)
            self.image_paths = valid_paths
            self.feature_dim = self.image_features.shape[1]
            
            print(f"Successfully built index with {len(valid_paths)} images")
            print(f"Feature dimension: {self.feature_dim}")
        else:
            print("No valid features extracted!")
    
    def save_index(self, save_path="./clip_image_index.pkl"):
        """
        保存图像索引库
        
        Args:
            save_path: 保存路径
        """
        if self.image_features.size == 0 or len(self.image_paths) == 0:
            print("No index to save!")
            return
        
        index_data = {
            'image_features': self.image_features,
            'image_paths': self.image_paths,
            'feature_dim': self.feature_dim
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Index saved to: {save_path}")
    
    def load_index(self, load_path="./clip_image_index.pkl"):
        """
        加载图像索引库
        
        Args:
            load_path: 加载路径
        """
        if not os.path.exists(load_path):
            print(f"Index file not found: {load_path}")
            return False
        
        with open(load_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.image_features = index_data['image_features']
        self.image_paths = index_data['image_paths']
        self.feature_dim = index_data['feature_dim']
        
        print(f"Index loaded from: {load_path}")
        print(f"Loaded {len(self.image_paths)} images with feature dim {self.feature_dim}")
        return True
    
    def search_similar_images(self, query_image_path, top_k=5):
        """
        搜索相似图像
        
        Args:
            query_image_path: 查询图像路径
            top_k: 返回最相似的K张图像
            
        Returns:
            list: 包含相似图像信息的列表
        """
        if self.image_features.size == 0 or len(self.image_paths) == 0:
            print("No image index available! Please build or load an index first.")
            return []
        
        # 提取查询图像特征
        query_features = self.extract_image_features(query_image_path)
        if query_features is None:
            print("Failed to extract features from query image!")
            return []
        
        # 计算余弦相似度
        similarities = cosine_similarity([query_features], self.image_features)[0]
        
        # 获取Top-K最相似的图像
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            result = {
                'rank': i + 1,
                'image_path': self.image_paths[idx],
                'similarity': similarities[idx],
                'filename': os.path.basename(self.image_paths[idx])
            }
            results.append(result)
        
        return results
    
    def visualize_search_results(self, query_image_path, results, save_path=None, figsize=(15, 10)):
        """
        可视化搜索结果
        
        Args:
            query_image_path: 查询图像路径
            results: 搜索结果列表
            save_path: 保存路径
            figsize: 图像大小
        """
        if not results:
            print("No results to visualize!")
            return
        
        # 计算子图布局
        n_results = len(results)
        cols = min(6, n_results + 1)  # +1 for query image
        rows = (n_results + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 显示查询图像
        query_image = Image.open(query_image_path).convert('RGB')
        
        ax = axes[0, 0]
        ax.imshow(query_image)
        ax.set_title(f'Query Image\n{os.path.basename(query_image_path)}', 
                    fontsize=12, fontweight='bold', color='red')
        ax.axis('off')
        
        # 添加红色边框
        rect = Rectangle((0, 0), query_image.width-1, query_image.height-1, 
                        linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # 显示搜索结果
        for i, result in enumerate(results):
            row = (i + 1) // cols
            col = (i + 1) % cols
            
            if row >= rows:
                break
                
            ax = axes[row, col]
            
            try:
                image = Image.open(result['image_path']).convert('RGB')
                ax.imshow(image)
                
                # 根据相似度设置标题颜色
                similarity = result['similarity']
                if similarity > 0.9:
                    color = 'green'
                elif similarity > 0.7:
                    color = 'orange'
                else:
                    color = 'black'
                
                ax.set_title(f'Rank {result["rank"]}\nSim: {similarity:.3f}\n{result["filename"]}', 
                           fontsize=10, color=color)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\n{result["filename"]}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Rank {result["rank"]} - Error', fontsize=10, color='red')
            
            ax.axis('off')
        
        # 隐藏剩余的子图
        for i in range(n_results + 1, rows * cols):
            row = i // cols
            col = i % cols
            if row < rows and col < cols:
                axes[row, col].axis('off')
        
        plt.suptitle('CLIP Image Retrieval Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def print_search_results(self, query_image_path, results):
        """
        打印搜索结果
        
        Args:
            query_image_path: 查询图像路径
            results: 搜索结果列表
        """
        print("\n" + "="*80)
        print("CLIP IMAGE RETRIEVAL RESULTS")
        print("="*80)
        print(f"Query Image: {os.path.basename(query_image_path)}")
        print(f"Found {len(results)} similar images:")
        print("-"*80)
        
        for result in results:
            print(f"Rank {result['rank']:2d}: {result['filename']:30s} | Similarity: {result['similarity']:.4f}")
        
        print("="*80)
    
    def batch_search(self, query_directory, top_k=5, save_results=True):
        """
        批量搜索查询目录中的所有图像
        
        Args:
            query_directory: 查询图像目录
            top_k: 每个查询返回的结果数
            save_results: 是否保存结果
            
        Returns:
            dict: 所有查询的结果
        """
        print(f"Starting batch search in: {query_directory}")
        
        # 收集查询图像
        query_files = []
        for file in os.listdir(query_directory):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                query_files.append(os.path.join(query_directory, file))
        
        if not query_files:
            print("No query images found!")
            return {}
        
        all_results = {}
        
        for i, query_path in enumerate(query_files):
            print(f"\nProcessing query {i+1}/{len(query_files)}: {os.path.basename(query_path)}")
            
            results = self.search_similar_images(query_path, top_k=top_k)
            if results:
                all_results[query_path] = results
                self.print_search_results(query_path, results)
                
                # 可视化结果
                vis_save_path = f"./results/batch_search_result_{i+1}.png"
                self.visualize_search_results(query_path, results, save_path=vis_save_path)
        
        if save_results and all_results:
            # 保存所有结果
            results_path = "./results/batch_search_results.json"
            os.makedirs("./results", exist_ok=True)
            
            # 转换为可序列化的格式
            serializable_results = {}
            for query_path, results in all_results.items():
                serializable_results[query_path] = [
                    {
                        'rank': r['rank'],
                        'image_path': r['image_path'],
                        'similarity': float(r['similarity']),
                        'filename': r['filename']
                    }
                    for r in results
                ]
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            print(f"\nBatch search results saved to: {results_path}")
        
        return all_results

def main():
    """主函数 - 演示CLIP图像检索系统的使用"""
    
    print("CLIP Image Retrieval System Demo")
    print("="*50)
    
    # 初始化检索系统
    retriever = CLIPImageRetrieval(model_name="ViT-B/32")
    
    # 设置数据路径
    image_database_dir = "../ResNet-50_Fine-Tuning/data"  # 图像数据库目录
    query_image_dir = "../ResNet-50_Fine-Tuning/data/test"  # 查询图像目录
    index_save_path = "./results/clip_image_index.pkl"
    
    # 检查数据目录是否存在
    if not os.path.exists(image_database_dir):
        print(f"Image database directory not found: {image_database_dir}")
        print("Please make sure you have images in the database directory.")
        return
    
    # 构建或加载图像索引
    if os.path.exists(index_save_path):
        print("Loading existing image index...")
        retriever.load_index(index_save_path)
    else:
        print("Building new image index...")
        retriever.build_image_index(image_database_dir)
        retriever.save_index(index_save_path)
    
    if len(retriever.image_paths) == 0:
        print("No images in the index! Please check your image directory.")
        return
    
    # 创建结果目录
    os.makedirs("./results", exist_ok=True)
    
    # 演示1: 单张图像搜索
    print("\n" + "="*50)
    print("Demo 1: Single Image Search")
    print("="*50)
    
    # 随机选择一张查询图像
    if os.path.exists(query_image_dir):
        query_files = [f for f in os.listdir(query_image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if query_files:
            import random
            # 随机选择一张图像
            selected_query = random.choice(query_files)
            query_image_path = os.path.join(query_image_dir, selected_query)
            print(f"Randomly selected query image: {query_image_path}")
            
            # 搜索相似图像
            results = retriever.search_similar_images(query_image_path, top_k=6)
            
            if results:
                # 打印结果
                retriever.print_search_results(query_image_path, results)
                
                # 可视化结果
                vis_save_path = "./results/single_search_result.png"
                retriever.visualize_search_results(query_image_path, results, save_path=vis_save_path)
            else:
                print("No search results found!")
        else:
            print(f"No query images found in: {query_image_dir}")
    
    # 演示2: 批量图像搜索
    print("\n" + "="*50)
    print("Demo 2: Batch Image Search")
    print("="*50)
    
    if os.path.exists(query_image_dir):
        # 随机选择5张图像进行批量搜索
        query_files = [f for f in os.listdir(query_image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if query_files:
            import random
            # 随机选择最多5张图像
            num_to_select = min(5, len(query_files))
            selected_files = random.sample(query_files, num_to_select)
            
            print(f"Found {len(query_files)} query images, randomly selected {num_to_select} for batch demo...")
            
            # 创建临时查询目录，只包含随机选择的图像
            temp_query_dir = "./results/temp_query"
            os.makedirs(temp_query_dir, exist_ok=True)
            
            for i, selected_file in enumerate(selected_files):
                src_path = os.path.join(query_image_dir, selected_file)
                dst_path = os.path.join(temp_query_dir, selected_file)
                
                # 复制文件
                from shutil import copy2
                copy2(src_path, dst_path)
                print(f"Selected image {i+1}: {selected_file}")
            
            batch_results = retriever.batch_search(temp_query_dir, top_k=5)
            
            # 清理临时目录
            import shutil
            shutil.rmtree(temp_query_dir)
        else:
            print(f"No query images found in: {query_image_dir}")
        
        print(f"\nBatch search completed! Processed {len(batch_results) if 'batch_results' in locals() else 0} queries.")
    else:
        print(f"Query directory not found: {query_image_dir}")
    
    # 演示3: 性能统计
    print("\n" + "="*50)
    print("Demo 3: System Statistics")
    print("="*50)
    
    print(f"Total images in database: {len(retriever.image_paths)}")
    print(f"Feature dimension: {retriever.feature_dim}")
    print(f"Index file size: {os.path.getsize(index_save_path) / 1024 / 1024:.2f} MB" 
          if os.path.exists(index_save_path) else "Index not saved")
    
    # 计算一些统计信息
    if len(retriever.image_features) > 0:
        feature_stats = {
            'mean': np.mean(retriever.image_features),
            'std': np.std(retriever.image_features),
            'min': np.min(retriever.image_features),
            'max': np.max(retriever.image_features)
        }
        
        print("\nFeature Statistics:")
        for key, value in feature_stats.items():
            print(f"  {key}: {value:.6f}")
    
    print("\n" + "="*50)
    print("CLIP Image Retrieval Demo Completed!")
    print("="*50)
    print("Check the './results/' directory for saved visualizations and data.")

if __name__ == "__main__":
    main()
