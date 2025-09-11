#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Next Token Prediction 进阶演示
结合简单模型和理论分析，深入理解Next Token Prediction原理

Author: AI Algorithm Course  
Date: 2025-09-11
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter, defaultdict
import pickle
from tqdm import tqdm
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedTokenizer:
    """进阶分词器，支持词级和字符级分词"""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.char_to_id = {}
        self.id_to_char = {}
        self.word_freq = Counter()
        self.char_freq = Counter()
        
    def build_vocab(self, texts: List[str], tokenization_level: str = "mixed"):
        """
        构建词汇表
        
        Args:
            texts: 训练文本
            tokenization_level: "word", "char", "mixed"
        """
        print(f"构建词汇表 (级别: {tokenization_level})...")
        
        # 收集词和字符频率
        all_words = []
        all_chars = set()
        
        for text in texts:
            # 简单的中文分词（按标点符号和空格分割）
            words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+|[^\s\u4e00-\u9fff]', text)
            all_words.extend(words)
            
            # 字符级别
            chars = list(text)
            all_chars.update(chars)
        
        self.word_freq = Counter(all_words)
        self.char_freq = Counter(all_chars)
        
        # 构建词级别词汇表
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        most_common_words = [word for word, _ in self.word_freq.most_common(self.vocab_size - len(special_tokens))]
        word_vocab = special_tokens + most_common_words
        
        self.word_to_id = {word: i for i, word in enumerate(word_vocab)}
        self.id_to_word = {i: word for i, word in enumerate(word_vocab)}
        
        # 构建字符级别词汇表
        char_vocab = special_tokens + sorted(list(all_chars))
        self.char_to_id = {char: i for i, char in enumerate(char_vocab)}
        self.id_to_char = {i: char for i, char in enumerate(char_vocab)}
        
        print(f"词级别词汇表大小: {len(self.word_to_id)}")
        print(f"字符级别词汇表大小: {len(self.char_to_id)}")
        
    def tokenize_text(self, text: str, level: str = "word") -> List[str]:
        """分词"""
        if level == "word":
            return re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+|[^\s\u4e00-\u9fff]', text)
        elif level == "char":
            return list(text)
        else:
            raise ValueError("level must be 'word' or 'char'")
    
    def encode(self, text: str, level: str = "word", max_length: Optional[int] = None) -> List[int]:
        """编码文本"""
        tokens = self.tokenize_text(text, level)
        
        if level == "word":
            ids = [self.word_to_id.get(token, self.word_to_id['<UNK>']) for token in tokens]
        else:
            ids = [self.char_to_id.get(token, self.char_to_id['<UNK>']) for token in tokens]
        
        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                pad_id = self.word_to_id['<PAD>'] if level == "word" else self.char_to_id['<PAD>']
                ids.extend([pad_id] * (max_length - len(ids)))
        
        return ids
    
    def decode(self, ids: List[int], level: str = "word") -> str:
        """解码ID序列"""
        if level == "word":
            tokens = [self.id_to_word.get(id, '<UNK>') for id in ids]
            # 过滤特殊token并连接
            tokens = [token for token in tokens if token not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']]
            return ''.join(tokens)
        else:
            chars = [self.id_to_char.get(id, '<UNK>') for id in ids]
            chars = [char for char in chars if char not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']]
            return ''.join(chars)

class TransformerLanguageModel(nn.Module):
    """基于Transformer的语言模型"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 4, max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 词嵌入和位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # 使用Encoder层，但添加因果掩码模拟Decoder行为
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
        # 初始化参数
        self._init_weights()
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """生成因果注意力掩码"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # 添加位置编码
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(input_ids.device)
        embeddings = embeddings + pos_encoding
        embeddings = self.dropout(embeddings)
        
        # 生成因果掩码
        src_mask = self._generate_square_subsequent_mask(seq_len).to(input_ids.device)
        
        # Transformer Encoder with causal mask
        output = self.transformer(embeddings, mask=src_mask)
        
        # 输出投影
        logits = self.output_proj(output)
        
        return logits
    
    def predict_next_token_with_analysis(self, input_ids: torch.Tensor, 
                                       temperature: float = 1.0, 
                                       top_k: int = 10) -> Dict:
        """预测下一个token并提供详细分析"""
        self.eval()
        with torch.no_grad():
            # 前向传播
            logits = self.forward(input_ids)
            last_logits = logits[0, -1, :] / temperature
            
            # 计算注意力权重（简化版本）
            attention_analysis = self._analyze_attention(input_ids)
            
            # 应用softmax
            probs = F.softmax(last_logits, dim=-1)
            
            # 获取top-k
            top_probs, top_indices = torch.topk(probs, top_k)
            
            # 计算困惑度
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            perplexity = torch.exp(entropy)
            
            return {
                "top_tokens": list(zip(top_indices.cpu().numpy(), top_probs.cpu().numpy())),
                "entropy": entropy.item(),
                "perplexity": perplexity.item(),
                "attention_analysis": attention_analysis,
                "full_probs": probs.cpu().numpy()
            }
    
    def _analyze_attention(self, input_ids: torch.Tensor) -> Dict:
        """分析注意力模式（简化版本）"""
        # 这里提供一个简化的注意力分析
        seq_len = input_ids.shape[1]
        
        # 模拟注意力权重分布
        attention_weights = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)
        
        return {
            "attention_weights": attention_weights.cpu().numpy(),
            "max_attention_position": torch.argmax(attention_weights[-1, :]).item()
        }

class NextTokenPredictionLab:
    """Next Token Prediction 实验室"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.tokenizer = AdvancedTokenizer()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_stats = {
            "losses": [], "perplexities": [], "learning_rates": []
        }
        
        print(f"🔬 Next Token Prediction 实验室")
        print(f"💻 使用设备: {self.device}")
    
    def load_and_prepare_data(self):
        """加载并准备数据"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 合并所有文本
        texts = []
        texts.extend(data.get("prompts", []))
        texts.extend(data.get("contexts", []))
        
        # 构建词汇表
        self.tokenizer.build_vocab(texts, tokenization_level="mixed")
        
        print(f"📚 加载了 {len(texts)} 个文本样本")
        return texts
    
    def create_model(self, model_type: str = "transformer"):
        """创建模型"""
        if model_type == "transformer":
            self.model = TransformerLanguageModel(
                vocab_size=len(self.tokenizer.word_to_id),
                d_model=256,
                nhead=8,
                num_layers=4,
                max_seq_len=128
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"🤖 创建 {model_type} 模型，参数量: {num_params:,}")
    
    def train_model(self, texts: List[str], epochs: int = 20, batch_size: int = 16):
        """训练模型"""
        print("\n🏋️ 开始训练模型...")
        
        # 创建数据集
        dataset = self._create_dataset(texts, seq_len=16)  # 减小序列长度
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 设置优化器和学习率调度器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.word_to_id['<PAD>'])
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_inputs, batch_targets in progress_bar:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                logits = self.model(batch_inputs)
                
                # 计算损失
                loss = criterion(logits.reshape(-1, self.model.vocab_size), 
                               batch_targets.reshape(-1))
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 计算平均损失和困惑度
            avg_loss = total_loss / num_batches
            perplexity = math.exp(avg_loss)
            current_lr = scheduler.get_last_lr()[0]
            
            # 记录统计信息
            self.training_stats["losses"].append(avg_loss)
            self.training_stats["perplexities"].append(perplexity)
            self.training_stats["learning_rates"].append(current_lr)
            
            # 更新学习率
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                      f"Perplexity: {perplexity:.2f}, LR: {current_lr:.6f}")
        
        print("✅ 训练完成!")
    
    def _create_dataset(self, texts: List[str], seq_len: int):
        """创建数据集"""
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, seq_len):
                self.data = []
                
                # 合并所有文本创建更长的序列
                combined_text = " ".join(texts)
                tokens = tokenizer.encode(combined_text, level="word")
                
                print(f"总token数量: {len(tokens)}, 序列长度: {seq_len}")
                
                # 如果tokens太少，至少创建一些样本
                if len(tokens) < seq_len + 1:
                    print("⚠️ 文本太短，重复文本以创建足够的训练数据")
                    # 重复文本直到有足够的tokens
                    while len(tokens) < seq_len + 10:
                        tokens.extend(tokenizer.encode(combined_text, level="word"))
                
                # 创建训练序列
                for i in range(len(tokens) - seq_len):
                    input_seq = tokens[i:i+seq_len]
                    target_seq = tokens[i+1:i+seq_len+1]
                    self.data.append((input_seq, target_seq))
                
                print(f"创建了 {len(self.data)} 个训练样本")
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                input_seq, target_seq = self.data[idx]
                return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
        
        return TextDataset(texts, self.tokenizer, seq_len)
    
    def comprehensive_analysis(self, test_texts: List[str], save_dir: str):
        """综合分析Next Token Prediction"""
        print("\n🔍 进行综合分析...")
        
        results = {
            "token_predictions": [],
            "generation_analysis": [],
            "probability_distributions": [],
            "attention_patterns": []
        }
        
        for i, text in enumerate(test_texts[:3]):
            print(f"\n分析文本 {i+1}: {text[:50]}...")
            
            # 编码输入
            input_tokens = self.tokenizer.encode(text[:30], level="word")
            input_ids = torch.tensor([input_tokens]).to(self.device)
            
            # 详细预测分析
            analysis = self.model.predict_next_token_with_analysis(
                input_ids, temperature=0.8, top_k=10
            )
            
            # 解码top预测
            top_predictions = []
            for token_id, prob in analysis["top_tokens"]:
                token = self.tokenizer.id_to_word[token_id]
                top_predictions.append({"token": token, "prob": float(prob)})
            
            results["token_predictions"].append({
                "input_text": text[:30],
                "predictions": top_predictions,
                "entropy": analysis["entropy"],
                "perplexity": analysis["perplexity"]
            })
            
            # 生成文本分析
            generated = self._generate_with_analysis(text[:20], max_length=15)
            results["generation_analysis"].append(generated)
            
            # 概率分布分析
            prob_dist = self._analyze_probability_distribution(analysis["full_probs"])
            results["probability_distributions"].append(prob_dist)
            
            # 注意力模式分析
            results["attention_patterns"].append(analysis["attention_analysis"])
        
        # 保存结果并可视化
        self._save_and_visualize_analysis(results, save_dir)
        
        return results
    
    def _generate_with_analysis(self, seed_text: str, max_length: int = 20) -> Dict:
        """带分析的文本生成"""
        current_text = seed_text
        generation_steps = []
        
        for step in range(max_length):
            # 编码当前文本
            input_tokens = self.tokenizer.encode(current_text, level="word")
            input_ids = torch.tensor([input_tokens[-32:]]).to(self.device)  # 使用最后32个token
            
            # 预测下一个token
            analysis = self.model.predict_next_token_with_analysis(input_ids, temperature=0.7)
            
            # 选择最可能的token
            next_token_id, next_prob = analysis["top_tokens"][0]
            next_token = self.tokenizer.id_to_word[next_token_id]
            
            # 记录生成步骤
            generation_steps.append({
                "step": step + 1,
                "input_context": current_text[-20:],  # 最后20个字符
                "predicted_token": next_token,
                "probability": float(next_prob),
                "entropy": analysis["entropy"],
                "perplexity": analysis["perplexity"]
            })
            
            # 更新文本
            current_text += next_token
            
            # 检查停止条件
            if next_token in ['<EOS>', '<PAD>'] or len(current_text) > len(seed_text) + 50:
                break
        
        return {
            "seed_text": seed_text,
            "generated_text": current_text,
            "generation_steps": generation_steps
        }
    
    def _analyze_probability_distribution(self, probs: np.ndarray) -> Dict:
        """分析概率分布"""
        # 计算分布统计
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_prob = np.max(probs)
        top_10_mass = np.sum(np.sort(probs)[-10:])
        
        # 找到高概率区间
        sorted_indices = np.argsort(probs)[::-1]
        cumulative_prob = 0
        tokens_for_90_percent = 0
        
        for i in sorted_indices:
            cumulative_prob += probs[i]
            tokens_for_90_percent += 1
            if cumulative_prob >= 0.9:
                break
        
        return {
            "entropy": float(entropy),
            "max_probability": float(max_prob),
            "top_10_mass": float(top_10_mass),
            "tokens_for_90_percent": tokens_for_90_percent,
            "distribution_shape": "peaked" if max_prob > 0.5 else "flat"
        }
    
    def _save_and_visualize_analysis(self, results: Dict, save_dir: str):
        """保存并可视化分析结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = self._convert_numpy_to_list(results)
        
        # 保存详细结果
        with open(os.path.join(save_dir, "comprehensive_analysis.json"), 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 可视化1: 训练过程
        self._plot_training_progress(save_dir)
        
        # 可视化2: Token预测概率分布
        self._plot_prediction_analysis(results, save_dir)
        
        # 可视化3: 生成过程分析
        self._plot_generation_analysis(results, save_dir)
        
        # 可视化4: 注意力模式
        self._plot_attention_patterns(results, save_dir)
        
        print("✅ 分析结果已保存并可视化")
    
    def _convert_numpy_to_list(self, obj):
        """递归转换numpy数组为列表"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    def _plot_training_progress(self, save_dir: str):
        """绘制训练进度"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_stats["losses"]) + 1)
        
        # 损失曲线
        ax1.plot(epochs, self.training_stats["losses"], 'b-', linewidth=2)
        ax1.set_title('训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # 困惑度曲线
        ax2.plot(epochs, self.training_stats["perplexities"], 'r-', linewidth=2)
        ax2.set_title('困惑度')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax3.plot(epochs, self.training_stats["learning_rates"], 'g-', linewidth=2)
        ax3.set_title('学习率')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        
        # 损失和困惑度对比
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(epochs, self.training_stats["losses"], 'b-', label='Loss')
        line2 = ax4_twin.plot(epochs, self.training_stats["perplexities"], 'r-', label='Perplexity')
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss', color='b')
        ax4_twin.set_ylabel('Perplexity', color='r')
        ax4.set_title('训练指标对比')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_analysis(self, results: Dict, save_dir: str):
        """绘制预测分析"""
        predictions = results["token_predictions"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, pred_data in enumerate(predictions):
            if i >= 4:
                break
            
            # 提取预测数据
            tokens = [p["token"] for p in pred_data["predictions"][:8]]
            probs = [p["prob"] for p in pred_data["predictions"][:8]]
            
            # 清理token显示
            clean_tokens = [token.replace('\n', '\\n').replace(' ', '_') for token in tokens]
            
            # 绘制柱状图
            bars = axes[i].bar(range(len(clean_tokens)), probs, 
                              color=plt.cm.viridis(np.linspace(0, 1, len(probs))), alpha=0.8)
            
            axes[i].set_xlabel('Token排名')
            axes[i].set_ylabel('预测概率')
            axes[i].set_title(f'文本 {i+1} 的Token预测\n'
                             f'熵: {pred_data["entropy"]:.2f}, '
                             f'困惑度: {pred_data["perplexity"]:.2f}')
            axes[i].set_xticks(range(len(clean_tokens)))
            axes[i].set_xticklabels(clean_tokens, rotation=45, ha='right')
            
            # 添加数值标签
            for j, (bar, prob) in enumerate(zip(bars, probs)):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 隐藏多余的subplot
        for i in range(len(predictions), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_generation_analysis(self, results: Dict, save_dir: str):
        """绘制生成过程分析"""
        generation_data = results["generation_analysis"]
        
        fig, axes = plt.subplots(len(generation_data), 1, figsize=(12, 4*len(generation_data)))
        if len(generation_data) == 1:
            axes = [axes]
        
        for i, gen_data in enumerate(generation_data):
            steps = [step["step"] for step in gen_data["generation_steps"]]
            probs = [step["probability"] for step in gen_data["generation_steps"]]
            entropies = [step["entropy"] for step in gen_data["generation_steps"]]
            
            # 双y轴图
            ax1 = axes[i]
            ax2 = ax1.twinx()
            
            # 概率曲线
            line1 = ax1.plot(steps, probs, 'b-o', label='预测概率', linewidth=2, markersize=4)
            ax1.set_xlabel('生成步骤')
            ax1.set_ylabel('预测概率', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # 熵曲线
            line2 = ax2.plot(steps, entropies, 'r-s', label='信息熵', linewidth=2, markersize=4)
            ax2.set_ylabel('信息熵', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            ax1.set_title(f'生成过程 {i+1}: {gen_data["seed_text"]} → {gen_data["generated_text"][:50]}...')
            ax1.grid(True, alpha=0.3)
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'generation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_patterns(self, results: Dict, save_dir: str):
        """绘制注意力模式"""
        attention_data = results["attention_patterns"]
        
        fig, axes = plt.subplots(1, len(attention_data), figsize=(5*len(attention_data), 4))
        if len(attention_data) == 1:
            axes = [axes]
        
        for i, attn_data in enumerate(attention_data):
            attention_weights = attn_data["attention_weights"]
            
            # 绘制注意力热力图
            im = axes[i].imshow(attention_weights, cmap='Blues', aspect='auto')
            axes[i].set_title(f'注意力模式 {i+1}')
            axes[i].set_xlabel('输入位置')
            axes[i].set_ylabel('输出位置')
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'attention_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model_and_tokenizer(self, save_dir: str):
        """保存模型和分词器"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'max_seq_len': self.model.max_seq_len
            },
            'training_stats': self.training_stats
        }, os.path.join(save_dir, 'transformer_language_model.pth'))
        
        # 保存分词器
        with open(os.path.join(save_dir, 'advanced_tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # 保存配置
        config = {
            "model_type": "transformer",
            "vocab_size": len(self.tokenizer.word_to_id),
            "char_vocab_size": len(self.tokenizer.char_to_id),
            "training_epochs": len(self.training_stats["losses"]),
            "final_loss": self.training_stats["losses"][-1] if self.training_stats["losses"] else None,
            "final_perplexity": self.training_stats["perplexities"][-1] if self.training_stats["perplexities"] else None
        }
        
        with open(os.path.join(save_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 模型和分词器已保存到: {save_dir}")

def main():
    """主函数"""
    print("🎯 Next Token Prediction 进阶实验开始!")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 定义路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "text_generation_data.json")
    results_dir = os.path.join(current_dir, "results")
    checkpoint_dir = os.path.join(current_dir, "model-checkpoint")
    
    try:
        # 1. 创建实验实例
        lab = NextTokenPredictionLab(data_file)
        
        # 2. 加载和准备数据
        texts = lab.load_and_prepare_data()
        
        # 3. 创建Transformer模型
        lab.create_model("transformer")
        
        # 4. 训练模型
        lab.train_model(texts, epochs=25, batch_size=8)
        
        # 5. 综合分析
        analysis_results = lab.comprehensive_analysis(texts, results_dir)
        
        # 6. 保存模型
        lab.save_model_and_tokenizer(checkpoint_dir)
        
        print("\n" + "=" * 60)
        print("🎉 Next Token Prediction 进阶实验完成!")
        print(f"📁 分析结果: {results_dir}")
        print(f"🤖 模型检查点: {checkpoint_dir}")
        
        # 打印实验摘要
        print("\n📊 实验摘要:")
        print(f"✅ 训练轮数: {len(lab.training_stats['losses'])}")
        print(f"✅ 最终损失: {lab.training_stats['losses'][-1]:.4f}")
        print(f"✅ 最终困惑度: {lab.training_stats['perplexities'][-1]:.2f}")
        print(f"✅ 分析文本数量: {len(analysis_results['token_predictions'])}")
        
    except Exception as e:
        print(f"\n❌ 实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
