"""
多模态CLIP模型训练主程序

该模块实现完整的训练流程，包括：
1. 数据加载和预处理
2. 模型初始化和配置
3. 训练循环和验证
4. 结果保存和可视化
5. 模型检查点管理

训练核心流程：
1. 加载预训练CLIP模型
2. 在自定义数据集上进行微调
3. 使用对比学习损失优化模型
4. 定期验证和保存最佳模型
"""

import os
import time
import json
import argparse
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from clip_model import CLIPModel, ContrastiveLoss, MultimodalTrainer
from clip_model import create_model_and_optimizer, create_scheduler
from data_loader import create_data_loaders, create_sample_dataset
from evaluation import ModelEvaluator
from visualization import TrainingVisualizer


class TrainingConfig:
    """训练配置类"""
    
    def __init__(self):
        # 数据配置
        self.data_dir = "./data"
        self.results_dir = "./results"
        self.checkpoint_dir = "./model-checkpoint"
        
        # 模型配置
        self.model_name = "ViT-B/32"  # 可选: ViT-B/16, ViT-L/14, RN50等
        self.projection_dim = 512
        self.freeze_backbone = False
        
        # 训练配置
        self.batch_size = 32
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.gradient_clip = 1.0
        
        # 验证和保存配置
        self.val_frequency = 1  # 每N个epoch验证一次
        self.save_frequency = 5  # 每N个epoch保存一次
        self.early_stopping_patience = 10  # 早停patience
        
        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4
        
        # 日志配置
        self.log_frequency = 10  # 每N个batch记录一次
        self.tensorboard_log = True
        
    def update_from_args(self, args):
        """从命令行参数更新配置"""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
                
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            
    @classmethod
    def load(cls, filepath: str):
        """从文件加载配置"""
        config = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        return config


def setup_training_environment(config: TrainingConfig) -> Tuple[str, SummaryWriter]:
    """
    设置训练环境
    
    Args:
        config: 训练配置
        
    Returns:
        (experiment_dir, writer): 实验目录和tensorboard writer
    """
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"clip_training_{timestamp}"
    experiment_dir = os.path.join(config.results_dir, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 保存配置
    config.save(os.path.join(experiment_dir, "config.json"))
    
    # 设置tensorboard
    writer = None
    if config.tensorboard_log:
        log_dir = os.path.join(experiment_dir, "logs")
        writer = SummaryWriter(log_dir)
        
    print(f"实验目录: {experiment_dir}")
    print(f"使用设备: {config.device}")
    
    return experiment_dir, writer


def train_epoch(trainer: MultimodalTrainer,
               train_loader,
               epoch: int,
               config: TrainingConfig,
               writer: SummaryWriter = None) -> Dict[str, float]:
    """
    训练一个epoch
    
    Args:
        trainer: 训练器
        train_loader: 训练数据加载器
        epoch: 当前epoch
        config: 训练配置
        writer: tensorboard writer
        
    Returns:
        epoch训练指标
    """
    trainer.model.train()
    
    # 用于累积指标
    epoch_metrics = {}
    total_batches = len(train_loader)
    
    # 进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for batch_idx, (images, texts) in enumerate(pbar):
        # 移动数据到设备
        images = images.to(config.device)
        texts = texts.to(config.device)
        
        # 训练步骤
        batch_metrics = trainer.train_step(images, texts)
        
        # 学习率调度
        if trainer.scheduler:
            trainer.scheduler.step()
            
        # 累积指标
        for key, value in batch_metrics.items():
            if key not in epoch_metrics:
                epoch_metrics[key] = []
            epoch_metrics[key].append(value)
            
        # 更新进度条
        current_loss = batch_metrics['total_loss']
        current_lr = trainer.optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f"{current_loss:.4f}",
            'lr': f"{current_lr:.2e}"
        })
        
        # 记录到tensorboard
        if writer and batch_idx % config.log_frequency == 0:
            step = epoch * total_batches + batch_idx
            writer.add_scalar('Train/Loss', current_loss, step)
            writer.add_scalar('Train/LearningRate', current_lr, step)
            
            for key, value in batch_metrics.items():
                if key != 'total_loss':
                    writer.add_scalar(f'Train/{key}', value, step)
    
    # 计算epoch平均指标
    avg_metrics = {}
    for key, values in epoch_metrics.items():
        avg_metrics[key] = np.mean(values)
        
    return avg_metrics


def validate_epoch(trainer: MultimodalTrainer,
                  val_loader,
                  epoch: int,
                  config: TrainingConfig,
                  writer: SummaryWriter = None) -> Dict[str, float]:
    """
    验证一个epoch
    
    Args:
        trainer: 训练器
        val_loader: 验证数据加载器
        epoch: 当前epoch
        config: 训练配置
        writer: tensorboard writer
        
    Returns:
        epoch验证指标
    """
    trainer.model.eval()
    
    # 用于累积指标
    epoch_metrics = {}
    
    # 进度条
    pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
    
    with torch.no_grad():
        for batch_idx, (images, texts) in enumerate(pbar):
            # 移动数据到设备
            images = images.to(config.device)
            texts = texts.to(config.device)
            
            # 验证步骤
            batch_metrics = trainer.validate_step(images, texts)
            
            # 累积指标
            for key, value in batch_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
                
            # 更新进度条
            current_loss = batch_metrics['total_loss']
            pbar.set_postfix({'val_loss': f"{current_loss:.4f}"})
    
    # 计算epoch平均指标
    avg_metrics = {}
    for key, values in epoch_metrics.items():
        avg_metrics[key] = np.mean(values)
        
    # 记录到tensorboard
    if writer:
        for key, value in avg_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
            
    return avg_metrics


def main_training_loop(config: TrainingConfig):
    """
    主训练循环
    
    Args:
        config: 训练配置
    """
    print("=" * 60)
    print("多模态CLIP模型训练开始")
    print("=" * 60)
    
    # 设置训练环境
    experiment_dir, writer = setup_training_environment(config)
    
    # 准备数据
    print("\n1. 准备数据...")
    if not os.path.exists(os.path.join(config.data_dir, 'data.json')):
        print("创建示例数据集...")
        create_sample_dataset(config.data_dir, num_samples=200)
        
    train_loader, val_loader = create_data_loaders(
        config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # 创建模型
    print("\n2. 初始化模型...")
    model, optimizer = create_model_and_optimizer(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        freeze_backbone=config.freeze_backbone,
        device=config.device
    )
    
    # 创建学习率调度器
    total_steps = len(train_loader) * config.num_epochs
    scheduler = create_scheduler(optimizer, total_steps, config.warmup_steps)
    
    # 创建训练器
    criterion = ContrastiveLoss()
    trainer = MultimodalTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=config.device
    )
    
    # 创建评估器和可视化器
    evaluator = ModelEvaluator(model, config.device)
    visualizer = TrainingVisualizer(experiment_dir)
    
    # 训练状态
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n3. 开始训练 (共{config.num_epochs}个epochs)...")
    start_time = time.time()
    
    try:
        for epoch in range(config.num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = train_epoch(trainer, train_loader, epoch, config, writer)
            trainer.train_history.append(train_metrics)
            
            # 验证
            if epoch % config.val_frequency == 0:
                val_metrics = validate_epoch(trainer, val_loader, epoch, config, writer)
                trainer.val_history.append(val_metrics)
                
                # 检查是否是最佳模型
                current_val_loss = val_metrics['total_loss']
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
                    trainer.save_checkpoint(
                        best_model_path, 
                        epoch, 
                        best_val_loss,
                        {'config': config.to_dict()}
                    )
                else:
                    patience_counter += 1
                    
                # 早停检查
                if patience_counter >= config.early_stopping_patience:
                    print(f"\n早停触发! 连续{config.early_stopping_patience}个epoch验证损失未改善")
                    break
                    
                # 打印epoch结果
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch+1}/{config.num_epochs} 完成 (用时: {epoch_time:.2f}s)")
                print(f"  训练损失: {train_metrics['total_loss']:.4f}")
                print(f"  验证损失: {current_val_loss:.4f} (最佳: {best_val_loss:.4f})")
                print(f"  图像->文本 Top1: {val_metrics['i2t_acc1']:.4f}")
                print(f"  文本->图像 Top1: {val_metrics['t2i_acc1']:.4f}")
                
            # 定期保存检查点
            if epoch % config.save_frequency == 0:
                checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                trainer.save_checkpoint(checkpoint_path, epoch, best_val_loss)
                
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        
    # 训练完成
    total_time = time.time() - start_time
    print(f"\n训练完成! 总用时: {total_time/3600:.2f}小时")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 最终评估
    print("\n4. 最终评估...")
    final_metrics = evaluator.comprehensive_evaluation(val_loader)
    
    # 保存最终结果
    results = {
        'config': config.to_dict(),
        'best_val_loss': best_val_loss,
        'total_training_time': total_time,
        'final_metrics': final_metrics,
        'train_history': trainer.train_history,
        'val_history': trainer.val_history
    }
    
    results_path = os.path.join(experiment_dir, "training_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    # 生成可视化
    print("\n5. 生成训练可视化...")
    visualizer.plot_training_curves(trainer.train_history, trainer.val_history)
    visualizer.plot_model_metrics(final_metrics)
    
    # 关闭tensorboard
    if writer:
        writer.close()
        
    print(f"\n所有结果已保存到: {experiment_dir}")
    print("训练完成!")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多模态CLIP模型训练")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data", 
                       help="数据目录路径")
    parser.add_argument("--results_dir", type=str, default="./results", 
                       help="结果保存目录")
    parser.add_argument("--checkpoint_dir", type=str, default="./model-checkpoint", 
                       help="模型检查点目录")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="ViT-B/32",
                       choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"],
                       help="预训练CLIP模型名称")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="是否冻结主干网络")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减")
    
    # 设备参数
    parser.add_argument("--device", type=str, default=None,
                       help="计算设备 (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载进程数")
    
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建配置
    config = TrainingConfig()
    config.update_from_args(args)
    
    # 如果没有指定设备，自动选择
    if config.device is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 开始训练
    main_training_loop(config)
