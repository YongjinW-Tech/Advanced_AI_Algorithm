#!/bin/bash

# Exercise-2 CLIP Image Retrieval Setup Script
# 安装CLIP图像检索系统所需的依赖

echo "==================================================="
echo "Setting up CLIP Image Retrieval System"
echo "==================================================="

# 检查Python环境
echo "Checking Python environment..."
python --version

# 安装基础依赖
echo "Installing basic dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装CLIP
echo "Installing OpenAI CLIP..."
pip install git+https://github.com/openai/CLIP.git

# 安装其他依赖
echo "Installing other requirements..."
pip install scikit-learn>=1.0.0
pip install Pillow>=9.0.0
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0

echo "==================================================="
echo "Setup completed!"
echo "==================================================="

# 验证安装
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import clip; print('CLIP: Successfully imported')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "import PIL; print('Pillow: Successfully imported')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"

echo "==================================================="
echo "All dependencies installed successfully!"
echo "You can now run: python Exercise-2_CLIP_Image_Retrieval.py"
echo "==================================================="
