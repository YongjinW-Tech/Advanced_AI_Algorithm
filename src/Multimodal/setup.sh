#!/bin/bash

# 多模态CLIP训练环境安装脚本

echo "开始安装多模态CLIP训练环境..."

# 检查Python版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装PyTorch (根据CUDA版本选择)
echo "安装PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU，安装CUDA版本的PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "未检测到NVIDIA GPU，安装CPU版本的PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装CLIP
echo "安装OpenAI CLIP..."
pip install git+https://github.com/openai/CLIP.git

# 安装其他依赖
echo "安装其他依赖包..."
pip install -r requirements.txt

# 验证安装
echo "验证安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import clip; print('CLIP安装成功')"

# 创建必要目录
echo "创建必要目录..."
mkdir -p data/images
mkdir -p data/captions
mkdir -p results
mkdir -p model-checkpoint
mkdir -p logs

echo "环境安装完成！"
echo ""
echo "使用方法："
echo "1. 准备图像-文本对数据到 ./data 目录"
echo "2. 运行训练: python train.py"
echo "3. 查看结果: ./results 目录"
