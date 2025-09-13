"""
模型评估模块

该模块提供多模态CLIP模型的评估功能，包括：
1. 图像-文本检索性能评估
2. 跨模态相似度分析
3. 模型表示质量评估
4. 多种评估指标计算
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from clip_model import CLIPModel


class ModelEvaluator:
    """
    模型评估器
    
    提供多种评估指标和可视化功能来评估CLIP模型的性能
    """
    
    def __init__(self, model: CLIPModel, device: str = "cuda"):
        """
        初始化评估器
        
        Args:
            model: 要评估的CLIP模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def compute_retrieval_metrics(self, 
                                image_features: torch.Tensor,
                                text_features: torch.Tensor) -> Dict[str, float]:
        """
        计算检索评估指标
        
        Args:
            image_features: 图像特征 [N, D]
            text_features: 文本特征 [N, D]
            
        Returns:
            包含各种检索指标的字典
        """
        batch_size = image_features.shape[0]
        
        # 计算相似度矩阵
        similarity_matrix = image_features @ text_features.T  # [N, N]
        
        # 图像到文本检索
        i2t_ranks = []
        for i in range(batch_size):
            # 获取第i个图像与所有文本的相似度
            sim_scores = similarity_matrix[i]  # [N]
            # 排序并获取排名
            sorted_indices = torch.argsort(sim_scores, descending=True)
            # 找到正确文本的排名
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0][0].item() + 1
            i2t_ranks.append(rank)
            
        # 文本到图像检索
        t2i_ranks = []
        for i in range(batch_size):
            # 获取第i个文本与所有图像的相似度
            sim_scores = similarity_matrix[:, i]  # [N]
            # 排序并获取排名
            sorted_indices = torch.argsort(sim_scores, descending=True)
            # 找到正确图像的排名
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0][0].item() + 1
            t2i_ranks.append(rank)
            
        # 计算各种指标
        i2t_ranks = np.array(i2t_ranks)
        t2i_ranks = np.array(t2i_ranks)
        
        metrics = {
            # 图像到文本检索
            'i2t_r1': np.mean(i2t_ranks <= 1) * 100,    # Recall@1
            'i2t_r5': np.mean(i2t_ranks <= 5) * 100,    # Recall@5
            'i2t_r10': np.mean(i2t_ranks <= 10) * 100,  # Recall@10
            'i2t_median_rank': np.median(i2t_ranks),
            'i2t_mean_rank': np.mean(i2t_ranks),
            
            # 文本到图像检索
            't2i_r1': np.mean(t2i_ranks <= 1) * 100,    # Recall@1
            't2i_r5': np.mean(t2i_ranks <= 5) * 100,    # Recall@5
            't2i_r10': np.mean(t2i_ranks <= 10) * 100,  # Recall@10
            't2i_median_rank': np.median(t2i_ranks),
            't2i_mean_rank': np.mean(t2i_ranks),
        }
        
        # 计算总体指标
        metrics['rsum'] = (metrics['i2t_r1'] + metrics['i2t_r5'] + metrics['i2t_r10'] + 
                          metrics['t2i_r1'] + metrics['t2i_r5'] + metrics['t2i_r10'])
        
        return metrics
    
    def evaluate_on_dataloader(self, dataloader) -> Dict[str, float]:
        """
        在数据加载器上评估模型
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            评估指标字典
        """
        all_image_features = []
        all_text_features = []
        
        print("收集特征用于评估...")
        with torch.no_grad():
            for images, texts in tqdm(dataloader):
                images = images.to(self.device)
                texts = texts.to(self.device)
                
                # 提取特征
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
                
        # 拼接所有特征
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        
        print(f"评估 {len(all_image_features)} 个样本...")
        
        # 计算检索指标
        metrics = self.compute_retrieval_metrics(all_image_features, all_text_features)
        
        return metrics
    
    def analyze_similarity_distribution(self, 
                                      image_features: torch.Tensor,
                                      text_features: torch.Tensor) -> Dict[str, float]:
        """
        分析相似度分布
        
        Args:
            image_features: 图像特征
            text_features: 文本特征
            
        Returns:
            相似度统计信息
        """
        # 计算相似度矩阵
        similarity_matrix = image_features @ text_features.T
        
        # 正样本相似度（对角线）
        positive_similarities = torch.diag(similarity_matrix)
        
        # 负样本相似度（非对角线）
        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool)
        negative_similarities = similarity_matrix[~mask]
        
        stats = {
            'positive_sim_mean': positive_similarities.mean().item(),
            'positive_sim_std': positive_similarities.std().item(),
            'negative_sim_mean': negative_similarities.mean().item(),
            'negative_sim_std': negative_similarities.std().item(),
            'similarity_gap': (positive_similarities.mean() - negative_similarities.mean()).item()
        }
        
        return stats
    
    def comprehensive_evaluation(self, dataloader) -> Dict[str, float]:
        """
        综合评估
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            完整的评估结果
        """
        print("开始综合评估...")
        
        # 基础检索评估
        retrieval_metrics = self.evaluate_on_dataloader(dataloader)
        
        # 收集特征用于进一步分析
        all_image_features = []
        all_text_features = []
        
        with torch.no_grad():
            for images, texts in dataloader:
                images = images.to(self.device)
                texts = texts.to(self.device)
                
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
                
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        
        # 相似度分析
        similarity_stats = self.analyze_similarity_distribution(
            all_image_features, all_text_features
        )
        
        # 合并所有指标
        comprehensive_metrics = {**retrieval_metrics, **similarity_stats}
        
        # 打印主要指标
        print("\n=== 评估结果 ===")
        print(f"图像→文本检索:")
        print(f"  R@1: {retrieval_metrics['i2t_r1']:.2f}%")
        print(f"  R@5: {retrieval_metrics['i2t_r5']:.2f}%")
        print(f"  R@10: {retrieval_metrics['i2t_r10']:.2f}%")
        
        print(f"文本→图像检索:")
        print(f"  R@1: {retrieval_metrics['t2i_r1']:.2f}%")
        print(f"  R@5: {retrieval_metrics['t2i_r5']:.2f}%")
        print(f"  R@10: {retrieval_metrics['t2i_r10']:.2f}%")
        
        print(f"总体指标:")
        print(f"  RSum: {retrieval_metrics['rsum']:.2f}")
        print(f"  相似度间隙: {similarity_stats['similarity_gap']:.4f}")
        
        return comprehensive_metrics


class CrossModalAnalyzer:
    """
    跨模态分析器
    
    提供跨模态表示学习的深度分析功能
    """
    
    def __init__(self, model: CLIPModel, device: str = "cuda"):
        self.model = model
        self.device = device
        
    def compute_cross_modal_alignment(self, 
                                    image_features: torch.Tensor,
                                    text_features: torch.Tensor) -> Dict[str, float]:
        """
        计算跨模态对齐程度
        
        Args:
            image_features: 图像特征
            text_features: 文本特征
            
        Returns:
            对齐指标
        """
        # 确保特征已归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算每对匹配样本的相似度
        paired_similarities = (image_features * text_features).sum(dim=-1)
        
        # 计算所有可能对的相似度
        all_similarities = image_features @ text_features.T
        
        # 对齐指标
        alignment_metrics = {
            'mean_paired_similarity': paired_similarities.mean().item(),
            'std_paired_similarity': paired_similarities.std().item(),
            'mean_cross_similarity': all_similarities.mean().item(),
            'alignment_score': paired_similarities.mean().item() - all_similarities.mean().item()
        }
        
        return alignment_metrics
    
    def visualize_feature_space(self, 
                               image_features: torch.Tensor,
                               text_features: torch.Tensor,
                               save_path: str = None,
                               method: str = "tsne") -> None:
        """
        可视化特征空间
        
        Args:
            image_features: 图像特征
            text_features: 文本特征
            save_path: 保存路径
            method: 降维方法 ("tsne" 或 "pca")
        """
        # 合并特征
        all_features = torch.cat([image_features, text_features], dim=0).numpy()
        
        # 创建标签
        n_images = len(image_features)
        n_texts = len(text_features)
        labels = ['Image'] * n_images + ['Text'] * n_texts
        colors = ['red'] * n_images + ['blue'] * n_texts
        
        # 降维
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
        else:  # pca
            reducer = PCA(n_components=2, random_state=42)
            
        features_2d = reducer.fit_transform(all_features)
        
        # 可视化
        plt.figure(figsize=(10, 8))
        
        # 分别绘制图像和文本特征
        img_indices = np.array(labels) == 'Image'
        txt_indices = np.array(labels) == 'Text'
        
        plt.scatter(features_2d[img_indices, 0], features_2d[img_indices, 1], 
                   c='red', alpha=0.6, label='Images', s=50)
        plt.scatter(features_2d[txt_indices, 0], features_2d[txt_indices, 1], 
                   c='blue', alpha=0.6, label='Texts', s=50)
        
        # 连接匹配的图像-文本对
        for i in range(min(n_images, n_texts)):
            plt.plot([features_2d[i, 0], features_2d[n_images + i, 0]], 
                    [features_2d[i, 1], features_2d[n_images + i, 1]], 
                    'gray', alpha=0.3, linewidth=0.5)
        
        plt.title(f'Cross-modal Feature Space Visualization ({method.upper()})')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征空间可视化已保存到: {save_path}")
        else:
            plt.show()
            
        plt.close()


if __name__ == "__main__":
    # 测试评估功能
    import torch
    from clip_model import CLIPModel, create_model_and_optimizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建模型
    model, _ = create_model_and_optimizer(device=device)
    
    # 创建评估器
    evaluator = ModelEvaluator(model, device)
    analyzer = CrossModalAnalyzer(model, device)
    
    # 创建模拟数据
    batch_size = 16
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    texts = torch.randint(0, 1000, (batch_size, 77)).to(device)
    
    # 提取特征
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
    
    # 测试检索评估
    print("测试检索评估...")
    retrieval_metrics = evaluator.compute_retrieval_metrics(
        image_features.cpu(), text_features.cpu()
    )
    
    for key, value in retrieval_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试相似度分析
    print("\n测试相似度分析...")
    similarity_stats = evaluator.analyze_similarity_distribution(
        image_features.cpu(), text_features.cpu()
    )
    
    for key, value in similarity_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试跨模态对齐分析
    print("\n测试跨模态对齐分析...")
    alignment_metrics = analyzer.compute_cross_modal_alignment(
        image_features.cpu(), text_features.cpu()
    )
    
    for key, value in alignment_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n评估模块测试完成!")
