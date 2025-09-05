# Exercise-2: CLIP 图像检索系统

## 项目简介

这个项目实现了基于 OpenAI CLIP 模型的图像检索系统（以图搜图），可以通过输入一张查询图像，从图像数据库中找到最相似的图像。

## 功能特性

### 核心功能
- ✅ **特征提取**: 使用 CLIP 模型提取图像的视觉特征向量
- ✅ **索引构建**: 构建高效的图像特征索引库
- ✅ **相似度搜索**: 基于余弦相似度的快速图像检索
- ✅ **可视化展示**: 直观展示查询图像和检索结果
- ✅ **批量处理**: 支持批量图像检索
- ✅ **索引保存/加载**: 支持索引的持久化存储

### 支持的CLIP模型
- `ViT-B/32` (默认) - 平衡性能和速度
- `ViT-B/16` - 更高精度
- `ViT-L/14` - 最高精度但需要更多计算资源

## 安装说明

### 方法1: 自动安装（推荐）

```bash
# 在 src 目录下运行
bash setu.sh
```

### 方法2: 手动安装

```bash
# 1. 安装 PyTorch
pip install torch torchvision

# 2. 安装 CLIP
pip install git+https://github.com/openai/CLIP.git

# 3. 安装其他依赖
pip install scikit-learn Pillow numpy matplotlib
```

### 方法3: 使用 requirements 文件

```bash
pip install -r requirements.txt
```

## 使用方法

### 快速开始

1. **准备图像数据**
   ```
   将图像放入数据目录，例如：
   ../ResNet-50_Fine-Tuning/data/
   ├── train/
   │   ├── positive/
   │   └── negative/
   └── test/
   ```

2. **运行简单演示**
   ```bash
   python simple_demo.py
   ```

3. **运行完整演示**
   ```bash
   python CLIP_Image_Retrieval.py
   ```

### 详细用法

#### 1. 初始化检索系统

```python
from CLIP_Image_Retrieval import CLIPImageRetrieval

# 初始化（可选择不同的CLIP模型）
retriever = CLIPImageRetrieval(model_name="ViT-B/32")
```

#### 2. 构建图像索引

```python
# 构建新索引
retriever.build_image_index("./path/to/image/directory")

# 保存索引
retriever.save_index("./clip_index.pkl")

# 加载已有索引
retriever.load_index("./clip_index.pkl")
```

#### 3. 搜索相似图像

```python
# 单张图像检索
results = retriever.search_similar_images("query_image.jpg", top_k=5)

# 打印结果
retriever.print_search_results("query_image.jpg", results)

# 可视化结果
retriever.visualize_search_results("query_image.jpg", results, 
                                  save_path="./result.png")
```

#### 4. 批量检索

```python
# 批量处理查询目录中的所有图像
batch_results = retriever.batch_search("./query_directory", top_k=5)
```

## 输出结果说明

### 检索结果格式

每个检索结果包含以下信息：
```python
{
    'rank': 1,                           # 排名
    'image_path': '/path/to/image.jpg',  # 图像路径
    'similarity': 0.8542,                # 余弦相似度 (0-1)
    'filename': 'image.jpg'              # 文件名
}
```

### 相似度分数解释

- **0.9 - 1.0**: 极高相似度（几乎相同）
- **0.7 - 0.9**: 高相似度（明显相关）
- **0.5 - 0.7**: 中等相似度（有一定关联）
- **0.0 - 0.5**: 低相似度（关联性较弱）

### 生成的文件

运行后会在 `./results/` 目录下生成：

- `clip_image_index.pkl` - 图像特征索引文件
- `single_search_result.png` - 单次检索可视化结果
- `batch_search_result_*.png` - 批量检索可视化结果
- `batch_search_results.json` - 批量检索详细结果
- `demo_result.png` - 演示结果图像

## 系统架构

```
CLIPImageRetrieval
├── __init__()                    # 初始化CLIP模型
├── extract_image_features()      # 提取单张图像特征
├── build_image_index()          # 构建图像索引库
├── save_index() / load_index()  # 索引持久化
├── search_similar_images()      # 相似图像搜索
├── visualize_search_results()   # 结果可视化
├── print_search_results()       # 打印结果
└── batch_search()              # 批量搜索
```

## 性能优化建议

### 1. 模型选择
- **快速原型**: 使用 `ViT-B/32`
- **生产环境**: 使用 `ViT-B/16` 或 `ViT-L/14`

### 2. 硬件配置
- **GPU**: 推荐使用CUDA加速
- **内存**: 大型数据库建议16GB+内存

### 3. 索引管理
- 预先构建索引并保存，避免重复计算
- 定期更新索引以包含新图像

## 常见问题

### Q1: 安装CLIP时出错怎么办？
```bash
# 尝试使用以下命令
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### Q2: GPU内存不足怎么办？
- 使用较小的CLIP模型 (`ViT-B/32`)
- 减少batch size
- 使用CPU模式

### Q3: 检索结果不准确怎么办？
- 确保图像质量良好
- 尝试不同的CLIP模型
- 检查图像预处理是否正确

### Q4: 如何处理大量图像？
- 使用索引缓存
- 分批处理图像
- 考虑使用更高效的向量数据库

## 扩展功能

### 可能的改进方向

1. **文本-图像检索**: 支持使用文本描述搜索图像
2. **多模态融合**: 结合图像和文本特征
3. **在线更新**: 支持动态添加新图像到索引
4. **分布式部署**: 支持大规模分布式图像检索
5. **Web界面**: 提供友好的Web交互界面

## 技术细节

### CLIP模型介绍
CLIP (Contrastive Language-Image Pre-training) 是OpenAI开发的多模态模型，能够理解图像和文本的关系，在图像检索任务中表现优异。

### 特征提取原理
1. 输入图像经过预处理（resize, normalize等）
2. 通过CLIP的视觉编码器提取特征向量
3. 对特征向量进行L2归一化
4. 使用余弦相似度计算图像间的相似性

### 索引结构
```python
{
    'image_features': np.array,  # 特征矩阵 [N, feature_dim]
    'image_paths': list,         # 图像路径列表
    'feature_dim': int          # 特征维度
}
```

## 许可证

本项目仅用于学习和研究目的。CLIP模型遵循OpenAI的许可证条款。

## 作者

YongjinW-Tech  
日期：2025年9月5日

---

如有问题或建议，请通过GitHub Issues反馈。
