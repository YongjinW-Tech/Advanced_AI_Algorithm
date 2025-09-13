# Next Token Prediction 使用说明

## 文件概览

本目录包含了完整的 Next Token Prediction 实验实现，掌握语言模型的核心预测机制。

## 核心文件

### 1. 主要代码文件

- **`3-nextTokenPrediction.py`** - 主实验文件（使用 OPT 预训练模型）
- **`3-nextTokenPrediction_simple.py`** - 简化演示版本（字符级 LSTM）
- **`3-nextTokenPrediction_advanced.py`** - 进阶版本（自训练 Transformer）

### 2. 数据文件

- **`text_generation_data.json`** - 训练数据（中文AI相关文本）
- **`config.json`** - 模型配置参数

### 3. 结果文件

- **`results/`** - 实验结果和可视化图表
- **`model-checkpoint/`** - 训练好的模型检查点
- **`Next_Token_Prediction_Report.md`** - 详细实验报告

## 快速开始

### 环境配置

```bash
# 激活虚拟环境
source ~/venvs/Advanced_AI_Algorithm/bin/activate

# 安装依赖（如果尚未安装）
pip install -r requirements.txt
```

### 运行实验

#### 1. 简化演示版本（推荐开始）
```bash
python 3-nextTokenPrediction_simple.py
```
- 使用字符级 LSTM 模型
- 训练时间短，易于理解
- 演示基本的 Next Token Prediction 机制

#### 2. 进阶 Transformer 版本
```bash
python 3-nextTokenPrediction_advanced.py
```
- 使用完整的 Transformer 架构
- 包含详细的分析和可视化
- 展示现代语言模型的核心技术

#### 3. 预训练模型版本（需要网络）
```bash
python 3-nextTokenPrediction.py
```
- 使用 HuggingFace 预训练模型
- 需要网络连接下载模型
- 展示工业级语言模型的能力

## 实验内容

### 1. 模型训练
- 数据预处理和分词
- 神经语言模型构建
- 训练过程优化

### 2. Next Token 预测
- 概率分布计算
- Top-k 预测分析
- 温度参数调节

### 3. 文本生成
- 自回归生成过程
- 不同采样策略
- 生成质量评估

### 4. 模型分析
- 注意力机制可视化
- 困惑度和熵的计算
- 训练过程监控

## 输出文件说明

### 可视化结果
- `training_progress.png` - 训练损失和困惑度变化
- `prediction_analysis.png` - Token 预测概率分布
- `generation_analysis.png` - 文本生成过程分析
- `attention_patterns.png` - 注意力权重可视化

### 数据文件
- `comprehensive_analysis.json` - 详细的分析数据
- `next_token_demo_results.json` - 演示结果数据

### 模型文件
- `transformer_language_model.pth` - Transformer 模型
- `simple_language_model.pth` - 简单 LSTM 模型
- `advanced_tokenizer.pkl` - 分词器

## 核心概念

### Next Token Prediction
语言模型的核心任务，通过预测序列中下一个最可能出现的词来学习语言规律。

### 关键技术
1. **自注意力机制** - 模型同时关注输入序列的所有位置
2. **因果掩码** - 确保模型只能看到当前位置之前的信息
3. **位置编码** - 为序列中的每个位置添加位置信息
4. **概率采样** - 从概率分布中采样下一个词

### 评估指标
- **困惑度 (Perplexity)** - 模型预测的不确定性
- **交叉熵损失** - 预测与真实标签的差距
- **信息熵** - 概率分布的不确定性度量

## 实验参数

### 模型配置
- 词汇表大小: 53 (词级) / 228 (字符级)
- 嵌入维度: 128-256
- 注意力头数: 8
- Transformer 层数: 4

### 训练配置
- 学习率: 1e-4 (初始)
- 批次大小: 8-16
- 训练轮数: 15-25
- 优化器: AdamW

## 扩展实验

### 1. 数据扩展
- 添加更多的中文文本数据
- 尝试不同领域的文本
- 增加数据预处理的复杂度

### 2. 模型改进
- 增加模型层数和参数量
- 尝试不同的注意力机制
- 实现更复杂的位置编码

### 3. 评估完善
- 加入 BLEU、ROUGE 评估指标
- 实现人工评估
- 对比不同模型的性能

## 常见问题

### Q: 训练时间太长怎么办？
A: 可以减少训练轮数、使用更小的模型或使用简化版本。

### Q: 生成的文本质量不高？
A: 这是正常的，受限于训练数据量和模型规模。可以尝试调整温度参数或使用更大的数据集。

### Q: 模型无法收敛？
A: 检查学习率设置，确保数据预处理正确，可以尝试更小的学习率。

### Q: 想使用预训练模型但网络连接失败？
A: 可以先使用简化版本或进阶版本，它们都是完全本地训练的。

## 技术支持

如果遇到问题，请检查：
1. Python 环境和依赖安装
2. 数据文件是否存在
3. 模型参数设置是否合理
4. 计算资源是否充足

---

**作者**: AI Algorithm Course  
**更新时间**: 2025年9月11日  
**版本**: v1.0
