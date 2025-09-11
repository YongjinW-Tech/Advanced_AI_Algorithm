#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Next Token Prediction 简化演示版本
使用本地训练的简单模型演示 Next Token Prediction 原理

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
import numpy as np
from typing import List, Dict, Tuple
import re
from collections import Counter, defaultdict
import pickle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimpleTokenizer:
    """简单的中文分词器"""
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts: List[str]):
        """构建词汇表"""
        # 收集所有字符
        all_chars = set()
        for text in texts:
            # 简单的字符级分词
            chars = list(text)
            all_chars.update(chars)
        
        # 添加特殊token
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        # 构建映射
        vocab = special_tokens + sorted(list(all_chars))
        self.char_to_id = {char: i for i, char in enumerate(vocab)}
        self.id_to_char = {i: char for i, char in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        print(f"词汇表大小: {self.vocab_size}")
        
    def encode(self, text: str, max_length: int = None) -> List[int]:
        """编码文本"""
        chars = list(text)
        ids = [self.char_to_id.get(char, self.char_to_id['<UNK>']) for char in chars]
        
        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids.extend([self.char_to_id['<PAD>']] * (max_length - len(ids)))
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """解码ID序列"""
        chars = [self.id_to_char.get(id, '<UNK>') for id in ids]
        # 移除特殊token
        chars = [char for char in chars if char not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']]
        return ''.join(chars)

class SimpleLanguageModel(nn.Module):
    """简单的语言模型用于演示Next Token Prediction"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, hidden=None):
        """前向传播"""
        # 词嵌入
        embeddings = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        embeddings = self.dropout(embeddings)
        
        # LSTM
        lstm_out, hidden = self.lstm(embeddings, hidden)  # [batch, seq_len, hidden_dim]
        lstm_out = self.dropout(lstm_out)
        
        # 输出投影
        logits = self.output_proj(lstm_out)  # [batch, seq_len, vocab_size]
        
        return logits, hidden
    
    def predict_next_token(self, input_ids, temperature=1.0, top_k=10):
        """预测下一个token"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids)
            # 取最后一个位置的logits
            last_logits = logits[0, -1, :] / temperature
            
            # 应用softmax
            probs = F.softmax(last_logits, dim=-1)
            
            # 获取top-k
            top_probs, top_indices = torch.topk(probs, top_k)
            
            return top_indices.cpu().numpy(), top_probs.cpu().numpy()

class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, seq_len: int = 50):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = []
        
        # 准备训练数据
        for text in texts:
            # 编码文本
            encoded = tokenizer.encode(text)
            
            # 创建输入-输出对
            for i in range(len(encoded) - seq_len):
                input_seq = encoded[i:i+seq_len]
                target_seq = encoded[i+1:i+seq_len+1]
                self.data.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

class NextTokenPredictionDemo:
    """Next Token Prediction 演示类"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_history = {"losses": []}
        
        print(f"使用设备: {self.device}")
    
    def load_data(self):
        """加载训练数据"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 合并所有文本数据
        texts = []
        texts.extend(data.get("prompts", []))
        texts.extend(data.get("contexts", []))
        
        print(f"加载了 {len(texts)} 个文本样本")
        return texts
    
    def prepare_model(self, texts: List[str]):
        """准备模型和分词器"""
        # 构建词汇表
        self.tokenizer.build_vocab(texts)
        
        # 创建模型
        self.model = SimpleLanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=128,
            hidden_dim=256,
            num_layers=2
        ).to(self.device)
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_model(self, texts: List[str], epochs: int = 10, batch_size: int = 32):
        """训练模型"""
        print("\n开始训练模型...")
        
        # 创建数据集
        dataset = TextDataset(texts, self.tokenizer, seq_len=30)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 设置优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.char_to_id['<PAD>'])
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_inputs, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                logits, _ = self.model(batch_inputs)
                
                # 计算损失
                loss = criterion(logits.reshape(-1, self.tokenizer.vocab_size), 
                               batch_targets.reshape(-1))
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            self.training_history["losses"].append(avg_loss)
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("训练完成!")
    
    def demonstrate_next_token_prediction(self, test_texts: List[str], save_dir: str):
        """演示Next Token预测"""
        print("\n演示Next Token预测...")
        
        results = {
            "predictions": [],
            "generation_examples": []
        }
        
        for i, text in enumerate(test_texts[:3]):
            # 使用文本的前一部分作为输入
            prefix = text[:len(text)//2]
            print(f"\n测试文本 {i+1}: {prefix}")
            
            # 编码输入
            input_ids = torch.tensor([self.tokenizer.encode(prefix[-20:])]).to(self.device)
            
            # 预测下一个token
            top_indices, top_probs = self.model.predict_next_token(input_ids, top_k=5)
            
            predictions = []
            print("Top-5 预测:")
            for j, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                char = self.tokenizer.id_to_char[idx]
                print(f"  {j+1}. '{char}' (概率: {prob:.4f})")
                predictions.append({"char": char, "prob": float(prob)})
            
            results["predictions"].append({
                "prefix": prefix,
                "predictions": predictions
            })
            
            # 生成文本示例
            generated = self.generate_text(prefix[-10:], max_length=20)
            print(f"生成文本: {generated}")
            
            results["generation_examples"].append({
                "prefix": prefix[-10:],
                "generated": generated
            })
        
        # 保存结果
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "next_token_demo_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def generate_text(self, seed_text: str, max_length: int = 30, temperature: float = 0.8):
        """生成文本"""
        self.model.eval()
        
        # 编码种子文本
        current_text = seed_text
        input_ids = self.tokenizer.encode(current_text)
        
        with torch.no_grad():
            for _ in range(max_length):
                # 准备输入
                input_tensor = torch.tensor([input_ids[-20:]]).to(self.device)  # 使用最后20个字符
                
                # 预测下一个token
                top_indices, top_probs = self.model.predict_next_token(input_tensor, temperature=temperature, top_k=3)
                
                # 随机采样
                probs = torch.tensor(top_probs)
                next_idx = torch.multinomial(probs, 1).item()
                next_token_id = top_indices[next_idx]
                
                # 添加到序列
                input_ids.append(next_token_id)
                
                # 解码检查是否应该停止
                next_char = self.tokenizer.id_to_char[next_token_id]
                if next_char in ['<EOS>', '<PAD>']:
                    break
                
                current_text += next_char
        
        return current_text
    
    def visualize_training_process(self, save_dir: str):
        """可视化训练过程"""
        if not self.training_history["losses"]:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history["losses"], linewidth=2)
        plt.title('训练损失变化')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 训练过程可视化已保存")
    
    def save_model(self, save_dir: str):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.tokenizer.vocab_size,
            'char_to_id': self.tokenizer.char_to_id,
            'id_to_char': self.tokenizer.id_to_char,
            'training_history': self.training_history
        }, os.path.join(save_dir, 'simple_language_model.pth'))
        
        # 保存分词器
        with open(os.path.join(save_dir, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        print(f"✅ 模型已保存到: {save_dir}")

def main():
    """主函数"""
    print("🎯 Next Token Prediction 演示开始!")
    print("=" * 50)
    
    # 定义路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "text_generation_data.json")
    results_dir = os.path.join(current_dir, "results")
    checkpoint_dir = os.path.join(current_dir, "model-checkpoint")
    
    try:
        # 创建演示实例
        demo = NextTokenPredictionDemo(data_file)
        
        # 加载数据
        texts = demo.load_data()
        
        # 准备模型
        demo.prepare_model(texts)
        
        # 训练模型
        demo.train_model(texts, epochs=15, batch_size=16)
        
        # 演示Next Token预测
        results = demo.demonstrate_next_token_prediction(texts, results_dir)
        
        # 可视化训练过程
        demo.visualize_training_process(results_dir)
        
        # 保存模型
        demo.save_model(checkpoint_dir)
        
        print("\n" + "=" * 50)
        print("🎉 Next Token Prediction 演示完成!")
        print(f"📁 结果保存在: {results_dir}")
        print(f"🤖 模型保存在: {checkpoint_dir}")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
