# Dependencies Analysis Report
# Advanced AI Algorithm 项目依赖分析报告

## 📊 项目文件概览

本项目包含以下8个Python文件：

1. **Exercise-1_1.py** - 用户画像数据生成和可视化
2. **Exercise-1_2.py** - 简单留出法数据分割
3. **Exercise-1_3.py** - 分层抽样算法实现
4. **Exercise-1_4.py** - 用户数据向量化与PCA降维
5. **Exercise-1_4_improved.py** - 改进版多种降维方法对比
6. **Exercise-1_5.py** - 综合聚类分析
7. **Hierarchical-sampling-algorithm.py** - 分层抽样算法（C/C++风格）
8. **Interview-Hierarchical-sampling.py** - 面试友好版分层抽样

## 🔍 依赖关系分析

### 核心依赖库分类：

#### 1. 数据处理和科学计算
- **pandas>=1.5.0** - 数据操作和分析
- **numpy>=1.21.0** - 数值计算

#### 2. 数据可视化
- **matplotlib>=3.5.0** - 基础绘图
- **seaborn>=0.11.0** - 统计图表

#### 3. 机器学习和数据预处理
- **scikit-learn>=1.0.0** - 包含以下模块：
  - `sklearn.decomposition.PCA` - 主成分分析
  - `sklearn.manifold.TSNE` - t-SNE降维
  - `sklearn.preprocessing.StandardScaler, RobustScaler, LabelEncoder` - 数据预处理
  - `sklearn.feature_selection.SelectKBest, f_classif` - 特征选择
  - `sklearn.model_selection.train_test_split` - 数据分割
  - `sklearn.cluster.KMeans` - K均值聚类
  - `sklearn.metrics.silhouette_score` - 聚类评估

#### 4. 深度学习和嵌入模型
- **sentence-transformers>=2.0.0** - BGE-m3文本嵌入模型
- **torch>=1.10.0** - PyTorch深度学习框架（sentence-transformers依赖）
- **transformers>=4.0.0** - Hugging Face变换器库

#### 5. 高级降维算法
- **umap-learn>=0.5.0** - UMAP非线性降维

#### 6. 其他工具库
- **tqdm>=4.64.0** - 进度条显示
- **typing-extensions>=4.0.0** - 类型提示扩展

### 内置库使用：
- `random` - 随机数生成
- `warnings` - 警告控制
- `typing` - 类型提示
- `collections.defaultdict` - 默认字典
- `subprocess` - 子进程管理

## 📦 安装建议

### 方法1：使用requirements.txt
```bash
pip install -r requirements.txt
```

### 方法2：手动安装核心依赖
```bash
# 基础依赖
pip install pandas numpy matplotlib seaborn scikit-learn

# 高级功能依赖
pip install sentence-transformers umap-learn

# 可选：GPU支持的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🔧 兼容性说明

- **Python版本**：建议Python 3.8+
- **操作系统**：支持Windows, macOS, Linux
- **内存需求**：建议8GB+（BGE-m3模型较大）
- **GPU支持**：可选，用于加速sentence-transformers

## ⚠️ 注意事项

1. **sentence-transformers**首次使用会自动下载BGE-m3模型（约2-3GB）
2. **umap-learn**需要编译，某些系统可能需要额外配置
3. 部分文件包含动态依赖安装逻辑（如Exercise-1_4_improved.py）
4. 所有可视化都配置了中文字体支持

## 📈 使用统计

- **最常用依赖**：pandas, numpy, matplotlib, seaborn, scikit-learn
- **高级功能**：sentence-transformers (文本嵌入), umap-learn (非线性降维)
- **总依赖数量**：8个主要第三方包
- **估计安装大小**：约3-4GB（包含预训练模型）
