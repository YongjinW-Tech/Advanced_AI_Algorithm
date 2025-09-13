"""
多模态CLIP模型定义

该模块实现了基于CLIP的多模态模型，包括：
1. 图像编码器（基于Vision Transformer或CNN）
2. 文本编码器（基于Transformer）
3. 跨模态对比学习损失函数
4. 模型微调和推理功能

CLIP核心思想：
- 使用对比学习训练图像和文本编码器
- 将图像和文本映射到同一向量空间
- 正样本对（匹配的图像-文本）相似度高，负样本对相似度低
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math


class CLIPModel(nn.Module):
    """
    多模态CLIP模型
    
    该类封装了CLIP模型的图像和文本编码器，并提供了训练和推理接口。
    支持从预训练CLIP模型初始化，然后在特定数据上微调。
    """
    
    def __init__(self, 
                 model_name: str = "ViT-B/32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 freeze_backbone: bool = False,
                 projection_dim: int = 512):
        """
        初始化CLIP模型
        
        Args:
            model_name: 预训练CLIP模型名称 (ViT-B/32, ViT-B/16, RN50等)
            device: 计算设备
            freeze_backbone: 是否冻结主干网络参数
            projection_dim: 投影层输出维度
        """
        super(CLIPModel, self).__init__()
        
        self.device = device
        self.projection_dim = projection_dim
        
        # 加载预训练CLIP模型
        print(f"加载预训练CLIP模型: {model_name}")
        self.clip_model, self.preprocess = clip.load(model_name, device=device)
        
        # 获取特征维度
        self.image_features_dim = self.clip_model.visual.output_dim
        self.text_features_dim = self.clip_model.transformer.width
        
        # 是否冻结预训练参数
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
                
        # 添加投影层进行微调
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_features_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_features_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        print(f"模型初始化完成:")
        print(f"  图像特征维度: {self.image_features_dim}")
        print(f"  文本特征维度: {self.text_features_dim}")
        print(f"  投影维度: {projection_dim}")
        print(f"  冻结主干网络: {freeze_backbone}")
        
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            images: 图像张量 [batch_size, 3, 224, 224]
            
        Returns:
            图像特征 [batch_size, projection_dim]
        """
        # 使用CLIP图像编码器
        with torch.no_grad() if hasattr(self, '_freeze_image') else torch.enable_grad():
            image_features = self.clip_model.encode_image(images)
            
        # 应用投影层
        image_features = self.image_projection(image_features.float())
        
        # L2归一化
        image_features = F.normalize(image_features, dim=-1)
        
        return image_features
    
    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        """
        编码文本
        
        Args:
            texts: 文本token张量 [batch_size, seq_len]
            
        Returns:
            文本特征 [batch_size, projection_dim]
        """
        # 使用CLIP文本编码器
        with torch.no_grad() if hasattr(self, '_freeze_text') else torch.enable_grad():
            text_features = self.clip_model.encode_text(texts)
            
        # 应用投影层
        text_features = self.text_projection(text_features.float())
        
        # L2归一化
        text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: 图像张量
            texts: 文本张量
            
        Returns:
            (image_features, text_features, logit_scale): 图像特征、文本特征和缩放因子
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        
        return image_features, text_features, self.logit_scale.exp()


class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数
    
    实现CLIP中使用的对比学习损失，包括：
    1. 图像到文本的交叉熵损失
    2. 文本到图像的交叉熵损失
    3. 对称损失的平均
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        初始化对比损失
        
        Args:
            temperature: 温度参数，控制softmax的锐度
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, 
                image_features: torch.Tensor, 
                text_features: torch.Tensor,
                logit_scale: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算对比损失
        
        Args:
            image_features: 图像特征 [batch_size, feature_dim]
            text_features: 文本特征 [batch_size, feature_dim]
            logit_scale: 缩放因子
            
        Returns:
            (loss, metrics): 损失值和评估指标
        """
        batch_size = image_features.shape[0]
        
        # 计算相似度矩阵
        # logits[i][j] = 第i个图像与第j个文本的相似度
        logits = logit_scale * image_features @ text_features.T
        
        # 创建标签：对角线为正样本（i图像对应i文本）
        labels = torch.arange(batch_size, device=image_features.device)
        
        # 计算图像到文本的损失
        loss_i2t = F.cross_entropy(logits, labels)
        
        # 计算文本到图像的损失
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # 总损失为两者的平均
        total_loss = (loss_i2t + loss_t2i) / 2
        
        # 计算准确率指标
        with torch.no_grad():
            # 图像到文本检索准确率
            i2t_acc1 = (logits.argmax(dim=1) == labels).float().mean()
            i2t_acc5 = (logits.topk(5, dim=1)[1] == labels.unsqueeze(1)).any(dim=1).float().mean()
            
            # 文本到图像检索准确率
            t2i_acc1 = (logits.T.argmax(dim=1) == labels).float().mean()
            t2i_acc5 = (logits.T.topk(5, dim=1)[1] == labels.unsqueeze(1)).any(dim=1).float().mean()
            
        metrics = {
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item(),
            'total_loss': total_loss.item(),
            'i2t_acc1': i2t_acc1.item(),
            'i2t_acc5': i2t_acc5.item(),
            't2i_acc1': t2i_acc1.item(),
            't2i_acc5': t2i_acc5.item(),
            'logit_scale': logit_scale.item()
        }
        
        return total_loss, metrics


class MultimodalTrainer:
    """
    多模态训练器
    
    封装了模型训练的完整流程，包括：
    1. 前向传播和损失计算
    2. 反向传播和参数更新
    3. 学习率调度
    4. 模型保存和加载
    """
    
    def __init__(self, 
                 model: CLIPModel,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 criterion: Optional[ContrastiveLoss] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化训练器
        
        Args:
            model: CLIP模型
            optimizer: 优化器
            scheduler: 学习率调度器
            criterion: 损失函数
            device: 计算设备
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or ContrastiveLoss()
        self.device = device
        
        # 训练历史
        self.train_history = []
        self.val_history = []
        
    def train_step(self, images: torch.Tensor, texts: torch.Tensor) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            images: 图像批次
            texts: 文本批次
            
        Returns:
            训练指标字典
        """
        self.model.train()
        
        # 前向传播
        image_features, text_features, logit_scale = self.model(images, texts)
        
        # 计算损失
        loss, metrics = self.criterion(image_features, text_features, logit_scale)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 参数更新
        self.optimizer.step()
        
        return metrics
    
    def validate_step(self, images: torch.Tensor, texts: torch.Tensor) -> Dict[str, float]:
        """
        单步验证
        
        Args:
            images: 图像批次
            texts: 文本批次
            
        Returns:
            验证指标字典
        """
        self.model.eval()
        
        with torch.no_grad():
            # 前向传播
            image_features, text_features, logit_scale = self.model(images, texts)
            
            # 计算损失
            loss, metrics = self.criterion(image_features, text_features, logit_scale)
            
        return metrics
    
    def save_checkpoint(self, 
                       filepath: str, 
                       epoch: int, 
                       best_val_loss: float,
                       additional_info: Optional[Dict] = None):
        """
        保存模型检查点
        
        Args:
            filepath: 保存路径
            epoch: 当前轮次
            best_val_loss: 最佳验证损失
            additional_info: 额外信息
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if additional_info:
            checkpoint.update(additional_info)
            
        torch.save(checkpoint, filepath)
        print(f"检查点已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """
        加载模型检查点
        
        Args:
            filepath: 检查点路径
            
        Returns:
            检查点信息字典
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        print(f"检查点已加载: {filepath}")
        print(f"恢复到第 {checkpoint['epoch']} 轮，最佳验证损失: {checkpoint['best_val_loss']:.4f}")
        
        return checkpoint


def create_model_and_optimizer(model_name: str = "ViT-B/32",
                              learning_rate: float = 1e-4,
                              weight_decay: float = 0.01,
                              freeze_backbone: bool = False,
                              device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[CLIPModel, torch.optim.Optimizer]:
    """
    创建模型和优化器
    
    Args:
        model_name: CLIP模型名称
        learning_rate: 学习率
        weight_decay: 权重衰减
        freeze_backbone: 是否冻结主干网络
        device: 计算设备
        
    Returns:
        (model, optimizer): 模型和优化器
    """
    # 创建模型
    model = CLIPModel(
        model_name=model_name,
        device=device,
        freeze_backbone=freeze_backbone
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6
    )
    
    return model, optimizer


def create_scheduler(optimizer: torch.optim.Optimizer,
                    num_training_steps: int,
                    warmup_steps: int = 1000) -> torch.optim.lr_scheduler._LRScheduler:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        num_training_steps: 总训练步数
        warmup_steps: 预热步数
        
    Returns:
        学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


if __name__ == "__main__":
    # 测试模型创建
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建模型
    model, optimizer = create_model_and_optimizer(device=device)
    
    # 创建损失函数
    criterion = ContrastiveLoss()
    
    # 测试前向传播
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    texts = torch.randint(0, 1000, (batch_size, 77)).to(device)  # 模拟tokenized文本
    
    with torch.no_grad():
        image_features, text_features, logit_scale = model(images, texts)
        loss, metrics = criterion(image_features, text_features, logit_scale)
        
    print(f"\n模型测试结果:")
    print(f"  图像特征形状: {image_features.shape}")
    print(f"  文本特征形状: {text_features.shape}")
    print(f"  损失值: {loss.item():.4f}")
    print(f"  logit_scale: {logit_scale.item():.4f}")
    
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
