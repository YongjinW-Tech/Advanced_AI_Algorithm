# 数据准备指南 - 图像-文本对数据

## 📊 数据格式要求

这个多模态CLIP训练项目支持两种主要的数据格式：

### 格式1：JSON索引文件（推荐）

创建一个 `data.json` 文件，包含所有图像-文本对的索引：

```json
[
  {
    "image_path": "images/cat_001.jpg",
    "caption": "一只橘色的猫坐在窗台上，阳光洒在它的毛发上"
  },
  {
    "image_path": "images/dog_002.jpg", 
    "caption": "一只金毛犬在公园里奔跑，背景是绿色的草地"
  },
  {
    "image_path": "images/sunset_003.jpg",
    "caption": "美丽的日落景色，天空呈现橙红色，海面波光粼粼"
  }
]
```

### 格式2：目录结构

```
data/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── image_003.jpg
└── captions/
    ├── image_001.txt
    ├── image_002.txt
    └── image_003.txt
```

每个txt文件包含对应图像的文本描述。

## 🖼️ 图像要求

### 基本要求
- **格式**: JPG, PNG, JPEG (推荐JPG)
- **尺寸**: 任意尺寸 (程序会自动resize到224×224)
- **颜色**: RGB彩色图像 (灰度图会自动转为RGB)
- **质量**: 建议清晰、无损的图像

### 推荐规格
- **分辨率**: 256×256以上 (避免过度压缩)
- **文件大小**: 每张100KB-2MB
- **数量**: 最少50对，推荐1000+对用于有效训练

### 图像内容建议
- **多样性**: 包含不同场景、物体、人物
- **清晰度**: 主体明确，背景不要过于复杂
- **质量**: 光线良好，色彩自然

## 📝 文本要求

### 描述内容
- **准确性**: 准确描述图像中的主要内容
- **详细程度**: 包含主体、动作、背景、颜色等信息
- **长度**: 建议10-50个词，不超过77个token

### 文本示例

**好的描述**：
```
"一只黑白相间的猫咪躺在红色沙发上，旁边有一个蓝色的抱枕"
"两个小朋友在海边玩沙子，背景是蔚蓝的大海和白云"
"现代简约风格的客厅，白色墙壁，木质茶几上放着绿色植物"
```

**避免的描述**：
```
"这是一张图片"  # 太简单
"复杂的室内场景包含了各种家具装饰品以及多个人物在进行不同的活动..." # 太复杂
"cat sofa"  # 太简短
```

## 📁 数据组织示例

让我为您创建一个完整的数据组织示例：

### 示例1：宠物数据集
```json
[
  {
    "image_path": "images/pets/cat_sleeping.jpg",
    "caption": "一只灰色的猫咪蜷缩在毛毯上睡觉"
  },
  {
    "image_path": "images/pets/dog_playing.jpg",
    "caption": "一只拉布拉多犬在草地上玩飞盘"
  },
  {
    "image_path": "images/pets/rabbit_eating.jpg",
    "caption": "白色的兔子在吃胡萝卜，坐在绿色的草地上"
  }
]
```

### 示例2：风景数据集
```json
[
  {
    "image_path": "images/landscapes/mountain_lake.jpg",
    "caption": "雪山倒映在清澈的湖水中，周围是茂密的森林"
  },
  {
    "image_path": "images/landscapes/beach_sunset.jpg",
    "caption": "金色的夕阳洒在沙滩上，海浪轻柔地拍打着岸边"
  },
  {
    "image_path": "images/landscapes/city_night.jpg",
    "caption": "城市夜景，高楼大厦灯火通明，街道上车流如织"
  }
]
```

## 🔨 数据准备工具

### 自动生成示例数据
如果您暂时没有准备数据，可以使用项目内置的示例数据生成功能：

```python
from data_loader import create_sample_dataset

# 生成100个示例样本
create_sample_dataset("./data", num_samples=100)
```

### 验证数据格式
```python
from data_loader import ImageTextDataset

# 测试数据加载
dataset = ImageTextDataset("./data")
print(f"数据集大小: {len(dataset)}")

# 查看第一个样本
image, text = dataset[0]
print(f"图像形状: {image.shape}")
print(f"文本形状: {text.shape}")
```

## 📋 数据质量检查清单

在开始训练前，请确保：

### ✅ 文件结构检查
- [ ] data.json文件存在且格式正确
- [ ] 所有图像文件路径正确
- [ ] 图像文件可以正常打开
- [ ] 文本描述不为空

### ✅ 内容质量检查
- [ ] 图像-文本对应关系正确
- [ ] 文本描述准确且有意义
- [ ] 数据量足够（推荐>100对）
- [ ] 内容多样性足够

### ✅ 技术规格检查
- [ ] 图像格式支持（JPG/PNG）
- [ ] 文本编码为UTF-8
- [ ] 文件路径使用相对路径
- [ ] 特殊字符正确处理

## 🎯 不同领域的数据建议

### 1. 电商产品
```json
{
  "image_path": "products/shoes_nike_001.jpg",
  "caption": "Nike白色运动鞋，黑色勾标，适合跑步和日常穿着"
}
```

### 2. 医学影像
```json
{
  "image_path": "medical/xray_chest_001.jpg", 
  "caption": "胸部X光片显示正常的肺部结构，心脏大小正常"
}
```

### 3. 艺术作品
```json
{
  "image_path": "art/painting_001.jpg",
  "caption": "印象派风格的风景画，描绘了春天的花园，色彩鲜明"
}
```

### 4. 食物图片
```json
{
  "image_path": "food/pizza_001.jpg",
  "caption": "意大利玛格丽特披萨，上面有新鲜罗勒叶和马苏里拉奶酪"
}
```

## 🚀 快速开始步骤

1. **创建数据目录结构**：
```bash
mkdir -p data/images
mkdir -p data/captions
```

2. **放置图像文件**：
将您的图像文件复制到 `data/images/` 目录

3. **创建文本描述**：
- 方式1：创建 `data/data.json` 文件
- 方式2：在 `data/captions/` 目录创建对应的txt文件

4. **验证数据**：
```bash
python3 -c "from data_loader import ImageTextDataset; d=ImageTextDataset('./data'); print(f'加载了{len(d)}个样本')"
```

5. **开始训练**：
```bash
python3 train.py
```

## 💡 数据准备技巧

### 1. 批量处理图像
```python
from PIL import Image
import os

def resize_images(input_dir, output_dir, size=(512, 512)):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(os.path.join(input_dir, filename))
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(os.path.join(output_dir, filename))
```

### 2. 批量生成描述模板
```python
def generate_caption_template(image_name):
    # 根据图像文件名生成基础描述模板
    base_name = os.path.splitext(image_name)[0]
    return f"这是一张关于{base_name}的图片"
```

### 3. 数据增强
```python
# 可以通过同一张图片生成多个不同角度的描述
captions = [
    "一只猫坐在窗台上",
    "窗台上有一只可爱的猫咪", 
    "阳光透过窗户照射在猫咪身上"
]
```

## 🛠️ 数据准备工具

项目提供了多个工具帮助您准备数据：

### 1. 示例数据生成器
```bash
# 生成示例数据用于测试
python3 prepare_data_example.py
```

### 2. 数据格式转换器
```bash
# 从CSV文件转换
python3 convert_data.py --format csv --input your_data.csv

# 从文件夹结构转换  
python3 convert_data.py --format folder --input ./your_images --captions-dir ./your_captions

# 从COCO格式转换
python3 convert_data.py --format coco --input annotations.json --images-dir ./images
```

### 3. 数据验证工具
```bash
# 验证数据完整性
python3 example_data/validate_data.py
```

## 📝 准备步骤总结

1. **选择数据来源**：
   - 自己拍摄/收集图像
   - 使用公开数据集
   - 网络爬虫收集（注意版权）

2. **整理数据格式**：
   - 使用提供的转换工具
   - 手动创建JSON文件
   - 按目录结构组织

3. **质量检查**：
   - 图像清晰度检查
   - 文本描述准确性
   - 数据完整性验证

4. **开始训练**：
   - 数据量：最少50对，推荐1000+
   - 运行：`python3 train.py --data_dir your_data_dir`

## 🎯 常见数据来源

- **公开数据集**：COCO, Flickr30k, Conceptual Captions
- **电商数据**：产品图片+描述
- **社交媒体**：图片+标题/评论
- **专业领域**：医学影像+报告，卫星图像+标注

现在您可以根据这个指南准备您的图像-文本对数据了！如果有任何问题，随时可以询问。
