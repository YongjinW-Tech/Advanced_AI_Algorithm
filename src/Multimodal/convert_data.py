#!/usr/bin/env python3
"""
数据转换工具

这个工具帮助您将各种格式的数据转换为项目需要的格式。
支持多种输入格式，自动生成标准的data.json文件。
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image


class DataConverter:
    """数据格式转换器"""
    
    def __init__(self, output_dir: str = "./converted_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "images").mkdir(exist_ok=True)
        print(f"📁 输出目录: {self.output_dir}")
    
    def from_csv(self, csv_file: str, image_col: str = "image", text_col: str = "caption") -> str:
        """
        从CSV文件转换数据
        
        CSV格式示例:
        image,caption
        path/to/image1.jpg,"一只猫坐在窗台上"
        path/to/image2.jpg,"美丽的日落景色"
        """
        print(f"📊 从CSV文件转换: {csv_file}")
        
        data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                if image_col not in row or text_col not in row:
                    print(f"⚠️  第{i+1}行缺少必要列: {image_col} 或 {text_col}")
                    continue
                
                image_path = row[image_col].strip()
                caption = row[text_col].strip()
                
                if not image_path or not caption:
                    print(f"⚠️  第{i+1}行数据为空")
                    continue
                
                # 复制图像文件
                new_image_path = self._copy_image(image_path, f"img_{i:04d}")
                if new_image_path:
                    data.append({
                        "image_path": new_image_path,
                        "caption": caption
                    })
        
        return self._save_json(data)
    
    def from_folder_structure(self, images_dir: str, captions_dir: str = None) -> str:
        """
        从文件夹结构转换数据
        
        结构1: 只有图像文件夹，根据文件名生成描述
        images/
        ├── cat_001.jpg
        ├── dog_002.jpg
        
        结构2: 图像和描述分别在不同文件夹
        images/       captions/
        ├── img1.jpg  ├── img1.txt
        ├── img2.jpg  ├── img2.txt
        """
        print(f"📁 从文件夹结构转换: {images_dir}")
        
        images_path = Path(images_dir)
        if not images_path.exists():
            raise FileNotFoundError(f"图像目录不存在: {images_dir}")
        
        data = []
        image_files = []
        
        # 获取所有图像文件
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(images_path.glob(ext))
        
        image_files.sort()
        
        for i, image_file in enumerate(image_files):
            # 复制图像
            new_image_path = self._copy_image(str(image_file), f"img_{i:04d}")
            if not new_image_path:
                continue
            
            # 获取描述
            caption = ""
            
            if captions_dir:
                # 从对应的文本文件读取
                caption_file = Path(captions_dir) / f"{image_file.stem}.txt"
                if caption_file.exists():
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                else:
                    print(f"⚠️  未找到描述文件: {caption_file}")
                    caption = self._generate_caption_from_filename(image_file.name)
            else:
                # 根据文件名生成描述
                caption = self._generate_caption_from_filename(image_file.name)
            
            if caption:
                data.append({
                    "image_path": new_image_path,
                    "caption": caption
                })
        
        return self._save_json(data)
    
    def from_coco_format(self, annotations_file: str, images_dir: str) -> str:
        """
        从COCO格式转换数据
        
        COCO注释格式:
        {
          "images": [{"id": 1, "file_name": "image1.jpg"}],
          "annotations": [{"image_id": 1, "caption": "description"}]
        }
        """
        print(f"🏷️  从COCO格式转换: {annotations_file}")
        
        with open(annotations_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 建立映射关系
        image_id_to_filename = {}
        for img in coco_data.get('images', []):
            image_id_to_filename[img['id']] = img['file_name']
        
        # 收集注释
        image_captions = {}
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            caption = ann.get('caption', '')
            
            if image_id not in image_captions:
                image_captions[image_id] = []
            image_captions[image_id].append(caption)
        
        # 生成数据
        data = []
        for image_id, captions in image_captions.items():
            if image_id not in image_id_to_filename:
                continue
            
            filename = image_id_to_filename[image_id]
            original_path = os.path.join(images_dir, filename)
            
            # 复制图像
            new_image_path = self._copy_image(original_path, f"coco_{image_id:06d}")
            if not new_image_path:
                continue
            
            # 使用第一个描述（或组合多个描述）
            if captions:
                caption = captions[0]  # 可以修改为组合多个描述
                data.append({
                    "image_path": new_image_path,
                    "caption": caption
                })
        
        return self._save_json(data)
    
    def _copy_image(self, source_path: str, base_name: str) -> str:
        """复制图像文件到输出目录"""
        source = Path(source_path)
        
        if not source.exists():
            print(f"⚠️  图像文件不存在: {source_path}")
            return None
        
        # 验证是否为有效图像
        try:
            with Image.open(source) as img:
                # 获取原始扩展名
                ext = source.suffix.lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    ext = '.jpg'
                
                # 生成新文件名
                new_filename = f"{base_name}{ext}"
                dest_path = self.output_dir / "images" / new_filename
                
                # 复制并可选地调整大小
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 如果图像过大，调整大小
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                img.save(dest_path, 'JPEG', quality=95)
                
                return f"images/{new_filename}"
                
        except Exception as e:
            print(f"❌ 图像处理失败 {source_path}: {e}")
            return None
    
    def _generate_caption_from_filename(self, filename: str) -> str:
        """根据文件名生成基础描述"""
        # 移除扩展名并清理文件名
        name = Path(filename).stem
        name = name.replace('_', ' ').replace('-', ' ')
        
        # 简单的关键词映射
        keywords = {
            'cat': '猫',
            'dog': '狗',
            'bird': '鸟',
            'flower': '花',
            'tree': '树',
            'car': '汽车',
            'house': '房子',
            'sunset': '日落',
            'mountain': '山',
            'lake': '湖',
            'beach': '海滩',
            'city': '城市',
            'food': '食物',
            'pizza': '披萨',
            'coffee': '咖啡'
        }
        
        # 查找关键词
        found_keywords = []
        name_lower = name.lower()
        for eng, chn in keywords.items():
            if eng in name_lower:
                found_keywords.append(chn)
        
        if found_keywords:
            return f"一张关于{','.join(found_keywords)}的图片"
        else:
            return f"一张名为'{name}'的图片"
    
    def _save_json(self, data: List[Dict]) -> str:
        """保存数据为JSON格式"""
        json_path = self.output_dir / "data.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 转换完成: {len(data)} 个样本")
        print(f"📄 JSON文件: {json_path}")
        
        # 生成统计信息
        self._generate_stats(data)
        
        return str(json_path)
    
    def _generate_stats(self, data: List[Dict]):
        """生成数据统计信息"""
        if not data:
            return
        
        total_chars = sum(len(item['caption']) for item in data)
        avg_length = total_chars / len(data)
        
        stats = {
            "总样本数": len(data),
            "平均描述长度": round(avg_length, 1),
            "最短描述": min(len(item['caption']) for item in data),
            "最长描述": max(len(item['caption']) for item in data),
            "输出目录": str(self.output_dir)
        }
        
        stats_path = self.output_dir / "conversion_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"📊 统计信息:")
        for key, value in stats.items():
            if key != "输出目录":
                print(f"  {key}: {value}")


def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(description="数据格式转换工具")
    parser.add_argument("--format", choices=["csv", "folder", "coco"], required=True,
                       help="输入数据格式")
    parser.add_argument("--input", required=True,
                       help="输入文件或目录路径")
    parser.add_argument("--output", default="./converted_data",
                       help="输出目录 (默认: ./converted_data)")
    parser.add_argument("--images-dir", 
                       help="图像目录 (用于COCO格式)")
    parser.add_argument("--captions-dir",
                       help="描述文件目录 (用于folder格式)")
    parser.add_argument("--image-col", default="image",
                       help="CSV中图像列名 (默认: image)")
    parser.add_argument("--text-col", default="caption", 
                       help="CSV中文本列名 (默认: caption)")
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = DataConverter(args.output)
    
    try:
        if args.format == "csv":
            json_path = converter.from_csv(args.input, args.image_col, args.text_col)
            
        elif args.format == "folder":
            json_path = converter.from_folder_structure(args.input, args.captions_dir)
            
        elif args.format == "coco":
            if not args.images_dir:
                print("❌ COCO格式需要指定 --images-dir 参数")
                return
            json_path = converter.from_coco_format(args.input, args.images_dir)
        
        print(f"\n🎉 转换成功完成!")
        print(f"📁 输出目录: {args.output}")
        print(f"📄 数据文件: {json_path}")
        print(f"\n🚀 下一步:")
        print(f"  python3 train.py --data_dir {args.output}")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")


if __name__ == "__main__":
    # 如果直接运行，显示使用示例
    import sys
    
    if len(sys.argv) == 1:
        print("🔧 数据转换工具使用示例:")
        print("")
        print("1. 从CSV文件转换:")
        print("   python3 convert_data.py --format csv --input data.csv")
        print("")
        print("2. 从文件夹结构转换:")
        print("   python3 convert_data.py --format folder --input ./images --captions-dir ./captions")
        print("")
        print("3. 从COCO格式转换:")
        print("   python3 convert_data.py --format coco --input annotations.json --images-dir ./images")
        print("")
        print("使用 --help 查看所有选项")
    else:
        main()
