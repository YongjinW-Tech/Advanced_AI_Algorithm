"""
训练可视化模块

该模块提供训练过程和结果的可视化功能，包括：
1. 训练曲线绘制（损失、准确率等）
2. 模型性能可视化
3. 跨模态特征空间可视化
4. 相似度矩阵热图
5. 检索结果展示
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import torch
from PIL import Image
import json

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('default')


class TrainingVisualizer:
    """
    训练可视化器
    
    提供训练过程的各种可视化功能
    """
    
    def __init__(self, save_dir: str):
        """
        初始化可视化器
        
        Args:
            save_dir: 图片保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_curves(self, 
                           train_history: List[Dict], 
                           val_history: List[Dict],
                           save_name: str = "training_curves.png") -> None:
        """
        绘制训练曲线
        
        Args:
            train_history: 训练历史
            val_history: 验证历史
            save_name: 保存文件名
        """
        if not train_history or not val_history:
            print("没有足够的训练历史数据用于绘制曲线")
            return
            
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # 提取数据
        train_epochs = list(range(len(train_history)))
        val_epochs = list(range(len(val_history)))
        
        # 1. 总损失
        train_losses = [h['total_loss'] for h in train_history]
        val_losses = [h['total_loss'] for h in val_history]
        
        axes[0, 0].plot(train_epochs, train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(val_epochs, val_losses, 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 图像到文本检索准确率
        train_i2t_acc1 = [h['i2t_acc1'] for h in train_history]
        val_i2t_acc1 = [h['i2t_acc1'] for h in val_history]
        
        axes[0, 1].plot(train_epochs, train_i2t_acc1, 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(val_epochs, val_i2t_acc1, 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Image→Text Accuracy@1')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 文本到图像检索准确率
        train_t2i_acc1 = [h['t2i_acc1'] for h in train_history]
        val_t2i_acc1 = [h['t2i_acc1'] for h in val_history]
        
        axes[0, 2].plot(train_epochs, train_t2i_acc1, 'b-', label='Train', linewidth=2)
        axes[0, 2].plot(val_epochs, val_t2i_acc1, 'r-', label='Validation', linewidth=2)
        axes[0, 2].set_title('Text→Image Accuracy@1')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 分类损失对比
        train_i2t_loss = [h['loss_i2t'] for h in train_history]
        train_t2i_loss = [h['loss_t2i'] for h in train_history]
        
        axes[1, 0].plot(train_epochs, train_i2t_loss, 'g-', label='Image→Text', linewidth=2)
        axes[1, 0].plot(train_epochs, train_t2i_loss, 'm-', label='Text→Image', linewidth=2)
        axes[1, 0].set_title('Contrastive Loss Components (Train)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Top-5准确率
        if 'i2t_acc5' in train_history[0]:
            train_i2t_acc5 = [h['i2t_acc5'] for h in train_history]
            val_i2t_acc5 = [h['i2t_acc5'] for h in val_history]
            
            axes[1, 1].plot(train_epochs, train_i2t_acc5, 'b-', label='Train I2T@5', linewidth=2)
            axes[1, 1].plot(val_epochs, val_i2t_acc5, 'r-', label='Val I2T@5', linewidth=2)
            
            if 't2i_acc5' in train_history[0]:
                train_t2i_acc5 = [h['t2i_acc5'] for h in train_history]
                val_t2i_acc5 = [h['t2i_acc5'] for h in val_history]
                
                axes[1, 1].plot(train_epochs, train_t2i_acc5, 'b--', label='Train T2I@5', linewidth=2)
                axes[1, 1].plot(val_epochs, val_t2i_acc5, 'r--', label='Val T2I@5', linewidth=2)
            
            axes[1, 1].set_title('Top-5 Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Logit Scale
        if 'logit_scale' in train_history[0]:
            train_logit_scale = [h['logit_scale'] for h in train_history]
            
            axes[1, 2].plot(train_epochs, train_logit_scale, 'purple', linewidth=2)
            axes[1, 2].set_title('Logit Scale')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Scale Value')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
        plt.close()
        
    def plot_model_metrics(self, 
                          metrics: Dict[str, float],
                          save_name: str = "model_metrics.png") -> None:
        """
        绘制模型评估指标
        
        Args:
            metrics: 评估指标字典
            save_name: 保存文件名
        """
        # 分类指标
        retrieval_metrics = {}
        similarity_metrics = {}
        
        for key, value in metrics.items():
            if any(x in key for x in ['r1', 'r5', 'r10', 'rsum', 'rank']):
                retrieval_metrics[key] = value
            elif any(x in key for x in ['sim', 'gap', 'alignment']):
                similarity_metrics[key] = value
                
        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 检索指标条形图
        if retrieval_metrics:
            # 选择主要的Recall指标
            recall_keys = [k for k in retrieval_metrics.keys() if 'r1' in k or 'r5' in k or 'r10' in k]
            if recall_keys:
                recall_values = [retrieval_metrics[k] for k in recall_keys]
                recall_labels = [k.replace('_', ' ').title() for k in recall_keys]
                
                bars = axes[0].bar(recall_labels, recall_values, 
                                 color=['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral', 'plum'])
                axes[0].set_title('Retrieval Performance (Recall %)', fontweight='bold')
                axes[0].set_ylabel('Recall (%)')
                axes[0].set_ylim(0, 100)
                
                # 添加数值标签
                for bar, value in zip(bars, recall_values):
                    height = bar.get_height()
                    axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                # 旋转x轴标签
                plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. 相似度指标
        if similarity_metrics:
            sim_keys = list(similarity_metrics.keys())
            sim_values = list(similarity_metrics.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(sim_keys)))
            bars = axes[1].bar(range(len(sim_keys)), sim_values, color=colors)
            axes[1].set_title('Similarity Metrics', fontweight='bold')
            axes[1].set_ylabel('Value')
            axes[1].set_xticks(range(len(sim_keys)))
            axes[1].set_xticklabels([k.replace('_', ' ').title() for k in sim_keys])
            
            # 添加数值标签
            for bar, value in zip(bars, sim_values):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 旋转x轴标签
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型指标图已保存到: {save_path}")
        plt.close()
        
    def plot_similarity_matrix(self, 
                             similarity_matrix: np.ndarray,
                             save_name: str = "similarity_matrix.png",
                             top_k: int = 20) -> None:
        """
        绘制相似度矩阵热图
        
        Args:
            similarity_matrix: 相似度矩阵 [N, N]
            save_name: 保存文件名
            top_k: 显示前k个样本
        """
        # 如果矩阵太大，只显示前k个样本
        if similarity_matrix.shape[0] > top_k:
            similarity_matrix = similarity_matrix[:top_k, :top_k]
            
        plt.figure(figsize=(12, 10))
        
        # 创建热图
        sns.heatmap(similarity_matrix, 
                   annot=True if top_k <= 10 else False,  # 只在小矩阵时显示数值
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Cosine Similarity'})
        
        plt.title(f'Image-Text Similarity Matrix (Top {top_k} samples)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Text Index')
        plt.ylabel('Image Index')
        
        # 保存图片
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"相似度矩阵热图已保存到: {save_path}")
        plt.close()
        
    def plot_retrieval_examples(self, 
                               images: List[np.ndarray],
                               texts: List[str],
                               similarities: np.ndarray,
                               save_name: str = "retrieval_examples.png",
                               num_examples: int = 5) -> None:
        """
        绘制检索示例
        
        Args:
            images: 图像列表
            texts: 文本列表
            similarities: 相似度矩阵
            save_name: 保存文件名
            num_examples: 显示的示例数量
        """
        fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
        
        for i in range(min(num_examples, len(images))):
            # 显示查询图像
            if len(images[i].shape) == 3:  # RGB图像
                axes[i, 0].imshow(images[i])
            else:  # 灰度图像
                axes[i, 0].imshow(images[i], cmap='gray')
            axes[i, 0].set_title(f'Query Image {i}')
            axes[i, 0].axis('off')
            
            # 显示对应文本
            axes[i, 1].text(0.1, 0.5, texts[i], fontsize=10, 
                           verticalalignment='center', wrap=True)
            axes[i, 1].set_title(f'Ground Truth Text')
            axes[i, 1].axis('off')
            
            # 找到最相似的文本
            sim_scores = similarities[i]
            top_indices = np.argsort(sim_scores)[::-1][:3]  # Top 3
            
            for j, idx in enumerate(top_indices[:2]):
                axes[i, 2+j].text(0.1, 0.5, f"Rank {j+1}:\n{texts[idx]}", 
                                 fontsize=9, verticalalignment='center', wrap=True)
                axes[i, 2+j].set_title(f'Retrieved Text (Sim: {sim_scores[idx]:.3f})')
                axes[i, 2+j].axis('off')
        
        plt.suptitle('Image-to-Text Retrieval Examples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"检索示例已保存到: {save_path}")
        plt.close()
        
    def plot_loss_components(self, 
                           train_history: List[Dict],
                           save_name: str = "loss_components.png") -> None:
        """
        绘制损失组件分析
        
        Args:
            train_history: 训练历史
            save_name: 保存文件名
        """
        if not train_history:
            return
            
        epochs = list(range(len(train_history)))
        
        # 提取不同损失组件
        total_loss = [h['total_loss'] for h in train_history]
        i2t_loss = [h.get('loss_i2t', 0) for h in train_history]
        t2i_loss = [h.get('loss_t2i', 0) for h in train_history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, total_loss, 'b-', linewidth=2, label='Total Loss')
        plt.title('Total Contrastive Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, i2t_loss, 'r-', linewidth=2, label='Image→Text')
        plt.plot(epochs, t2i_loss, 'g-', linewidth=2, label='Text→Image')
        plt.title('Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        if len(total_loss) > 10:  # 只在有足够数据时计算移动平均
            window = min(5, len(total_loss) // 4)
            smoothed_loss = np.convolve(total_loss, np.ones(window)/window, mode='valid')
            smooth_epochs = epochs[:len(smoothed_loss)]
            plt.plot(smooth_epochs, smoothed_loss, 'purple', linewidth=2)
            plt.title(f'Smoothed Total Loss (Window={window})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # 损失改善率
        loss_improvement = []
        for i in range(1, len(total_loss)):
            improvement = (total_loss[i-1] - total_loss[i]) / total_loss[i-1] * 100
            loss_improvement.append(improvement)
            
        plt.plot(epochs[1:], loss_improvement, 'orange', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Loss Improvement Rate (%)')
        plt.xlabel('Epoch')
        plt.ylabel('Improvement Rate (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失组件分析图已保存到: {save_path}")
        plt.close()
        
    def create_training_summary(self, 
                              config: Dict,
                              final_metrics: Dict,
                              train_history: List[Dict],
                              val_history: List[Dict]) -> None:
        """
        创建训练总结报告
        
        Args:
            config: 训练配置
            final_metrics: 最终评估指标
            train_history: 训练历史
            val_history: 验证历史
        """
        # 创建总结文档
        summary = {
            "实验配置": {
                "模型": config.get('model_name', 'ViT-B/32'),
                "批次大小": config.get('batch_size', 32),
                "学习率": config.get('learning_rate', 1e-4),
                "训练轮数": len(train_history),
                "设备": config.get('device', 'unknown')
            },
            "最终性能": {
                "图像→文本 R@1": f"{final_metrics.get('i2t_r1', 0):.2f}%",
                "图像→文本 R@5": f"{final_metrics.get('i2t_r5', 0):.2f}%",
                "文本→图像 R@1": f"{final_metrics.get('t2i_r1', 0):.2f}%",
                "文本→图像 R@5": f"{final_metrics.get('t2i_r5', 0):.2f}%",
                "总和得分": f"{final_metrics.get('rsum', 0):.2f}",
                "相似度间隙": f"{final_metrics.get('similarity_gap', 0):.4f}"
            },
            "训练过程": {
                "最终训练损失": f"{train_history[-1]['total_loss']:.4f}" if train_history else "N/A",
                "最终验证损失": f"{val_history[-1]['total_loss']:.4f}" if val_history else "N/A",
                "最佳训练准确率": f"{max([h['i2t_acc1'] for h in train_history]):.4f}" if train_history else "N/A",
                "最佳验证准确率": f"{max([h['i2t_acc1'] for h in val_history]):.4f}" if val_history else "N/A"
            }
        }
        
        # 保存为JSON
        summary_path = os.path.join(self.save_dir, "training_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        print(f"训练总结已保存到: {summary_path}")


if __name__ == "__main__":
    # 测试可视化功能
    save_dir = "./test_viz"
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建可视化器
    visualizer = TrainingVisualizer(save_dir)
    
    # 创建模拟训练历史
    train_history = []
    val_history = []
    
    for epoch in range(20):
        # 模拟训练指标（损失递减，准确率递增）
        train_metrics = {
            'total_loss': 2.0 - epoch * 0.05 + np.random.normal(0, 0.02),
            'loss_i2t': 1.0 - epoch * 0.02 + np.random.normal(0, 0.01),
            'loss_t2i': 1.0 - epoch * 0.03 + np.random.normal(0, 0.01),
            'i2t_acc1': 0.1 + epoch * 0.03 + np.random.normal(0, 0.01),
            'i2t_acc5': 0.3 + epoch * 0.02 + np.random.normal(0, 0.01),
            't2i_acc1': 0.08 + epoch * 0.035 + np.random.normal(0, 0.01),
            't2i_acc5': 0.25 + epoch * 0.025 + np.random.normal(0, 0.01),
            'logit_scale': 14 + epoch * 0.1 + np.random.normal(0, 0.2)
        }
        
        # 验证指标（每2个epoch一次）
        if epoch % 2 == 0:
            val_metrics = {
                'total_loss': train_metrics['total_loss'] + 0.1,
                'loss_i2t': train_metrics['loss_i2t'] + 0.05,
                'loss_t2i': train_metrics['loss_t2i'] + 0.05,
                'i2t_acc1': train_metrics['i2t_acc1'] - 0.02,
                'i2t_acc5': train_metrics['i2t_acc5'] - 0.01,
                't2i_acc1': train_metrics['t2i_acc1'] - 0.02,
                't2i_acc5': train_metrics['t2i_acc5'] - 0.01,
                'logit_scale': train_metrics['logit_scale']
            }
            val_history.append(val_metrics)
            
        train_history.append(train_metrics)
    
    # 测试训练曲线绘制
    print("测试训练曲线绘制...")
    visualizer.plot_training_curves(train_history, val_history)
    
    # 测试模型指标绘制
    print("测试模型指标绘制...")
    final_metrics = {
        'i2t_r1': 45.6, 'i2t_r5': 78.2, 'i2t_r10': 89.1,
        't2i_r1': 42.3, 't2i_r5': 75.8, 't2i_r10': 86.7,
        'rsum': 317.7,
        'positive_sim_mean': 0.65, 'negative_sim_mean': 0.23,
        'similarity_gap': 0.42
    }
    visualizer.plot_model_metrics(final_metrics)
    
    # 测试相似度矩阵
    print("测试相似度矩阵绘制...")
    similarity_matrix = np.random.rand(10, 10)
    # 让对角线值更高（模拟正确匹配）
    for i in range(10):
        similarity_matrix[i, i] += 0.5
    visualizer.plot_similarity_matrix(similarity_matrix)
    
    # 测试损失组件分析
    print("测试损失组件分析...")
    visualizer.plot_loss_components(train_history)
    
    print(f"所有测试图片已保存到: {save_dir}")
    print("可视化模块测试完成!")
