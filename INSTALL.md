# 🚀 环境配置指南

## 📦 快速安装

### 方法1：自动配置（推荐）
```bash
python setup_environment.py
```

### 方法2：使用requirements.txt
```bash
pip install -r requirements.txt
```

### 方法3：手动安装
```bash
pip install pandas numpy matplotlib seaborn scikit-learn sentence-transformers umap-learn tqdm
```

## 🔍 验证安装
```bash
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, sentence_transformers, umap, tqdm; print('✅ 所有包导入成功!')"
```

## 📋 依赖列表

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| pandas | >=1.5.0 | 数据操作和分析 |
| numpy | >=1.21.0 | 数值计算 |
| matplotlib | >=3.5.0 | 基础绘图 |
| seaborn | >=0.11.0 | 统计图表 |
| scikit-learn | >=1.0.0 | 机器学习算法 |
| sentence-transformers | >=2.0.0 | 文本嵌入模型 |
| umap-learn | >=0.5.0 | UMAP降维算法 |
| tqdm | >=4.64.0 | 进度条显示 |

## ⚠️ 注意事项

1. **首次运行**：sentence-transformers会自动下载BGE-m3模型（约2-3GB）
2. **Python版本**：建议Python 3.8+
3. **内存要求**：建议8GB+RAM
4. **网络连接**：首次安装和运行需要稳定的网络连接

## 🐛 故障排除

### 常见问题

1. **umap-learn安装失败**
   ```bash
   # macOS
   brew install llvm
   
   # Ubuntu/Debian
   sudo apt-get install build-essential
   
   # Windows
   # 安装Visual Studio Build Tools
   ```

2. **sentence-transformers下载慢**
   ```bash
   # 设置HuggingFace镜像
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **CUDA支持（可选）**
   ```bash
   # GPU版本PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 📊 使用验证

安装完成后，您可以运行项目中的任意Python文件来验证环境配置：

```bash
cd src/
python Exercise-1_1.py  # 用户数据生成
python Exercise-1_4_improved.py  # 高级降维分析
```

## 📚 详细信息

更多技术细节请参考 [DEPENDENCIES.md](DEPENDENCIES.md) 文件。
