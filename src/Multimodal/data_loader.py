"""
多模态数据加载和预处理模块

该模块负责处理图像-文本对数据，为CLIP模型训练提供数据支持。
主要功能包括：
1. 加载图像-文本对数据
2. 图像预处理（resize, normalize等）
3. 文本预处理（tokenization等）
4. 创建PyTorch DataLoader
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import clip
from typing import List, Tuple, Dict


class ImageTextDataset(Dataset):
    """
    图像-文本对数据集类
    
    用于加载和处理图像-文本对数据，支持CLIP模型的训练。
    每个样本包含一张图像及其对应的文本描述。
    """
    
    def __init__(self, 
                 data_dir: str, 
                 image_transform=None, 
                 text_tokenizer=None,
                 max_text_length: int = 77):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            image_transform: 图像预处理transforms
            text_tokenizer: 文本tokenizer
            max_text_length: 文本最大长度（CLIP默认为77）
        """
        self.data_dir = data_dir
        self.max_text_length = max_text_length
        
        # 设置图像预处理
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # CLIP使用224x224输入
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP预训练参数
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            self.image_transform = image_transform
            
        # 设置文本tokenizer
        if text_tokenizer is None:
            # 使用CLIP的tokenizer
            self.text_tokenizer = clip.tokenize
        else:
            self.text_tokenizer = text_tokenizer
            
        # 加载数据
        self.data_pairs = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """
        加载图像-文本对数据
        
        支持多种数据格式：
        1. JSON文件格式：包含image_path和caption字段
        2. 目录结构：images/和captions/目录分别存放图像和文本文件
        
        Returns:
            包含图像路径和文本描述的字典列表
        """
        data_pairs = []
        
        # 尝试加载JSON格式数据
        json_file = os.path.join(self.data_dir, 'data.json')
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data_pairs = json.load(f)
        else:
            # 尝试从目录结构加载
            images_dir = os.path.join(self.data_dir, 'images')
            captions_dir = os.path.join(self.data_dir, 'captions')
            
            if os.path.exists(images_dir) and os.path.exists(captions_dir):
                image_files = sorted([f for f in os.listdir(images_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                for img_file in image_files:
                    # 查找对应的文本文件
                    base_name = os.path.splitext(img_file)[0]
                    caption_file = os.path.join(captions_dir, f"{base_name}.txt")
                    
                    if os.path.exists(caption_file):
                        with open(caption_file, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        
                        data_pairs.append({
                            'image_path': os.path.join(images_dir, img_file),
                            'caption': caption
                        })
            else:
                # 创建示例数据
                print(f"未找到数据文件，创建示例数据...")
                data_pairs = self._create_sample_data()
                
        print(f"加载了 {len(data_pairs)} 个图像-文本对")
        return data_pairs
    
    def _create_sample_data(self) -> List[Dict]:
        """
        创建示例数据用于演示
        
        Returns:
            示例图像-文本对列表
        """
        # 创建示例数据目录
        sample_data = []
        
        # 这里可以添加一些示例图像和描述
        # 实际使用时应该准备真实的图像-文本对数据
        sample_captions = [
            "A beautiful sunset over the ocean",
            "A cat sitting on a window sill",
            "A person riding a bicycle in the park",
            "Mountains covered with snow",
            "A delicious pizza with various toppings"
        ]
        
        for i, caption in enumerate(sample_captions):
            sample_data.append({
                'image_path': f'sample_image_{i}.jpg',  # 占位符，实际需要真实图像
                'caption': caption
            })
            
        # 保存示例数据
        sample_file = os.path.join(self.data_dir, 'sample_data.json')
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
        print(f"已创建示例数据文件: {sample_file}")
        return sample_data
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (image_tensor, text_tensor): 预处理后的图像和文本张量
        """
        data_pair = self.data_pairs[idx]
        image_path = data_pair['image_path']
        caption = data_pair['caption']
        
        # 加载和预处理图像
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.image_transform(image)
            except Exception as e:
                print(f"加载图像失败 {image_path}: {e}")
                # 创建随机图像作为占位符
                image_tensor = torch.randn(3, 224, 224)
        else:
            # 如果图像不存在，创建随机图像
            image_tensor = torch.randn(3, 224, 224)
            
        # 预处理文本
        try:
            text_tensor = self.text_tokenizer(caption, truncate=True).squeeze(0)
        except:
            # 如果tokenization失败，使用空白token
            text_tensor = torch.zeros(self.max_text_length, dtype=torch.long)
            
        return image_tensor, text_tensor


def create_data_loaders(data_dir: str, 
                       batch_size: int = 32,
                       train_split: float = 0.8,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小
        train_split: 训练集比例
        num_workers: 数据加载进程数
        
    Returns:
        (train_loader, val_loader): 训练和验证数据加载器
    """
    # 创建完整数据集
    full_dataset = ImageTextDataset(data_dir)
    
    # 分割训练和验证集
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"创建数据加载器完成:")
    print(f"  训练集大小: {train_size}")
    print(f"  验证集大小: {val_size}")
    print(f"  批次大小: {batch_size}")
    
    return train_loader, val_loader


def create_sample_dataset(data_dir: str, num_samples: int = 100):
    """
    创建示例数据集用于测试
    
    Args:
        data_dir: 数据保存目录
        num_samples: 样本数量
    """
    import random
    
    # 确保目录存在
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'captions'), exist_ok=True)
    
    # 示例描述模板
    caption_templates = [
        "A photo of a {object} in the {location}",
        "A beautiful {object} with {attribute}",
        "{number} {object} on a {surface}",
        "A {color} {object} during {time}",
        "Someone {action} with a {object}"
    ]
    
    objects = ['cat', 'dog', 'car', 'tree', 'house', 'flower', 'bird', 'mountain']
    locations = ['park', 'garden', 'street', 'forest', 'beach', 'field']
    attributes = ['bright colors', 'amazing details', 'perfect lighting']
    surfaces = ['table', 'ground', 'chair', 'bed', 'grass']
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
    times = ['sunset', 'sunrise', 'night', 'daytime']
    actions = ['playing', 'walking', 'running', 'sitting', 'standing']
    numbers = ['one', 'two', 'three', 'several', 'many']
    
    data_pairs = []
    
    for i in range(num_samples):
        # 随机选择模板和填充词
        template = random.choice(caption_templates)
        caption = template.format(
            object=random.choice(objects),
            location=random.choice(locations),
            attribute=random.choice(attributes),
            surface=random.choice(surfaces),
            color=random.choice(colors),
            time=random.choice(times),
            action=random.choice(actions),
            number=random.choice(numbers)
        )
        
        # 创建随机图像（实际应用中应使用真实图像）
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        image_path = os.path.join(data_dir, 'images', f'sample_{i:04d}.png')
        image.save(image_path)
        
        # 保存文本描述
        caption_path = os.path.join(data_dir, 'captions', f'sample_{i:04d}.txt')
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption)
            
        data_pairs.append({
            'image_path': image_path,
            'caption': caption
        })
    
    # 保存数据索引
    data_file = os.path.join(data_dir, 'data.json')
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(data_pairs, f, ensure_ascii=False, indent=2)
        
    print(f"已创建 {num_samples} 个示例样本到 {data_dir}")


if __name__ == "__main__":
    # 测试数据加载器
    data_dir = "./data"
    
    # 创建示例数据集
    print("创建示例数据集...")
    create_sample_dataset(data_dir, num_samples=50)
    
    # 测试数据加载
    print("\n测试数据加载...")
    train_loader, val_loader = create_data_loaders(data_dir, batch_size=8)
    
    # 查看一个批次的数据
    for images, texts in train_loader:
        print(f"图像批次形状: {images.shape}")
        print(f"文本批次形状: {texts.shape}")
        break
