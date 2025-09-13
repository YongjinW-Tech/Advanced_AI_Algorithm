#!/usr/bin/env python3
"""
数据准备示例脚本

这个脚本展示了如何准备真实的图像-文本对数据用于CLIP训练。
包含了常见的数据处理场景和最佳实践。
"""

import os
import json
import shutil
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple


def create_real_world_example():
    """创建真实世界的数据准备示例"""
    
    print("🎯 创建真实世界数据准备示例...")
    
    # 示例数据：不同领域的图像-文本对
    example_data = [
        # 宠物类
        {
            "category": "pets",
            "examples": [
                {
                    "filename": "cat_sleeping.jpg",
                    "caption": "一只橘色的猫咪蜷缩在蓝色毛毯上睡觉，阳光从窗户洒进来"
                },
                {
                    "filename": "dog_running.jpg", 
                    "caption": "一只金毛犬在绿色草地上奔跑，嘴里叼着红色飞盘"
                },
                {
                    "filename": "rabbit_garden.jpg",
                    "caption": "白色兔子在花园里吃胡萝卜，周围有紫色的花朵"
                }
            ]
        },
        # 风景类
        {
            "category": "landscapes",
            "examples": [
                {
                    "filename": "mountain_lake.jpg",
                    "caption": "雪山倒映在清澈的高山湖泊中，湖边有茂密的松树林"
                },
                {
                    "filename": "sunset_beach.jpg",
                    "caption": "金色夕阳西下，海浪轻柔地拍打着沙滩，天空呈现橙红色"
                },
                {
                    "filename": "autumn_forest.jpg",
                    "caption": "秋天的森林小径，两旁是金黄色的枫叶，地面铺满落叶"
                }
            ]
        },
        # 食物类
        {
            "category": "food",
            "examples": [
                {
                    "filename": "pizza_margherita.jpg",
                    "caption": "意大利玛格丽特披萨，上面有新鲜罗勒叶、马苏里拉奶酪和番茄酱"
                },
                {
                    "filename": "sushi_plate.jpg",
                    "caption": "精美的寿司拼盘，包含三文鱼、金枪鱼和鳗鱼寿司，配有芥末和酱油"
                },
                {
                    "filename": "coffee_croissant.jpg",
                    "caption": "一杯热腾腾的拿铁咖啡和金黄色的牛角包，放在木质桌面上"
                }
            ]
        },
        # 城市类
        {
            "category": "urban",
            "examples": [
                {
                    "filename": "city_skyline.jpg",
                    "caption": "现代都市天际线，高楼大厦在夜晚灯火通明，倒映在河面上"
                },
                {
                    "filename": "street_cafe.jpg",
                    "caption": "欧式街边咖啡馆，红色遮阳伞下坐着喝咖啡的人们"
                },
                {
                    "filename": "subway_station.jpg",
                    "caption": "繁忙的地铁站台，人们在等待列车，现代化的建筑设计"
                }
            ]
        }
    ]
    
    return example_data


def prepare_data_structure(base_dir: str = "./example_data"):
    """准备数据目录结构"""
    
    print(f"📁 创建数据目录结构: {base_dir}")
    
    # 创建主要目录
    dirs_to_create = [
        f"{base_dir}/images/pets",
        f"{base_dir}/images/landscapes", 
        f"{base_dir}/images/food",
        f"{base_dir}/images/urban",
        f"{base_dir}/captions/pets",
        f"{base_dir}/captions/landscapes",
        f"{base_dir}/captions/food", 
        f"{base_dir}/captions/urban"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✅ {dir_path}")
    
    return base_dir


def create_placeholder_images(data_structure: List[Dict], base_dir: str):
    """创建占位符图像（实际使用时替换为真实图像）"""
    
    print("🖼️  创建占位符图像...")
    
    colors = {
        'pets': (255, 200, 150),      # 暖橙色
        'landscapes': (100, 200, 100), # 绿色
        'food': (255, 150, 100),       # 橙红色
        'urban': (100, 150, 200)       # 蓝色
    }
    
    all_data = []
    
    for category_data in data_structure:
        category = category_data['category']
        base_color = colors.get(category, (128, 128, 128))
        
        for i, example in enumerate(category_data['examples']):
            filename = example['filename']
            caption = example['caption']
            
            # 创建占位符图像
            image_array = np.ones((224, 224, 3), dtype=np.uint8)
            
            # 添加一些随机变化
            np.random.seed(hash(filename) % 2**32)
            noise = np.random.randint(-30, 30, (224, 224, 3))
            
            for c in range(3):
                image_array[:, :, c] = np.clip(base_color[c] + noise[:, :, c], 0, 255)
            
            # 添加一些几何图案来区分不同图像
            center_x, center_y = 112, 112
            radius = 50 + i * 10
            y, x = np.ogrid[:224, :224]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            if mask.any():
                image_array[mask] = [255 - base_color[0], 255 - base_color[1], 255 - base_color[2]]
            
            # 保存图像
            image = Image.fromarray(image_array)
            image_path = f"{base_dir}/images/{category}/{filename}"
            image.save(image_path)
            
            # 保存文本描述
            caption_path = f"{base_dir}/captions/{category}/{os.path.splitext(filename)[0]}.txt"
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            # 记录数据
            all_data.append({
                "image_path": f"images/{category}/{filename}",
                "caption": caption,
                "category": category
            })
            
            print(f"  ✅ {category}/{filename}")
    
    return all_data


def create_data_json(all_data: List[Dict], base_dir: str):
    """创建data.json索引文件"""
    
    print("📄 创建data.json索引文件...")
    
    # 创建标准格式的数据索引
    json_data = []
    for item in all_data:
        json_data.append({
            "image_path": item["image_path"],
            "caption": item["caption"]
        })
    
    # 保存JSON文件
    json_path = f"{base_dir}/data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ 保存到: {json_path}")
    print(f"  📊 总计: {len(json_data)} 个图像-文本对")
    
    return json_path


def create_statistics_report(all_data: List[Dict], base_dir: str):
    """创建数据统计报告"""
    
    print("📊 生成数据统计报告...")
    
    # 统计各类别数量
    category_counts = {}
    total_caption_length = 0
    
    for item in all_data:
        category = item.get('category', 'unknown')
        category_counts[category] = category_counts.get(category, 0) + 1
        total_caption_length += len(item['caption'])
    
    # 生成报告
    report = {
        "数据总览": {
            "总样本数": len(all_data),
            "平均描述长度": round(total_caption_length / len(all_data), 1),
            "类别分布": category_counts
        },
        "数据质量": {
            "描述语言": "中文",
            "图像格式": "JPG",
            "图像尺寸": "224x224",
            "编码格式": "UTF-8"
        },
        "使用建议": {
            "最小训练样本": "50-100对",
            "推荐训练样本": "1000+对", 
            "最大文本长度": "77 tokens",
            "推荐描述长度": "10-30词"
        }
    }
    
    # 保存报告
    report_path = f"{base_dir}/data_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 打印报告
    print(f"  📈 数据统计:")
    print(f"    总样本数: {report['数据总览']['总样本数']}")
    print(f"    平均描述长度: {report['数据总览']['平均描述长度']} 字符")
    print(f"    类别分布:")
    for category, count in report['数据总览']['类别分布'].items():
        print(f"      {category}: {count} 个")
    
    return report_path


def create_data_validation_script(base_dir: str):
    """创建数据验证脚本"""
    
    validation_script = f'''#!/usr/bin/env python3
"""
数据验证脚本 - 自动生成

验证 {base_dir} 目录中的数据质量和完整性
"""

import os
import json
from PIL import Image

def validate_data():
    base_dir = "{base_dir}"
    
    print("🔍 验证数据完整性...")
    
    # 检查JSON文件
    json_path = os.path.join(base_dir, "data.json")
    if not os.path.exists(json_path):
        print("❌ data.json 文件不存在")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 加载了 {{len(data)}} 个数据样本")
    
    # 验证每个样本
    valid_count = 0
    for i, item in enumerate(data):
        image_path = os.path.join(base_dir, item['image_path'])
        caption = item['caption']
        
        # 检查图像文件
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {{image_path}}")
            continue
        
        # 检查图像是否可以打开
        try:
            with Image.open(image_path) as img:
                if img.size[0] == 0 or img.size[1] == 0:
                    print(f"❌ 图像尺寸无效: {{image_path}}")
                    continue
        except Exception as e:
            print(f"❌ 图像打开失败: {{image_path}} - {{e}}")
            continue
        
        # 检查文本描述
        if not caption or len(caption.strip()) == 0:
            print(f"❌ 文本描述为空: index {{i}}")
            continue
        
        valid_count += 1
    
    print(f"✅ 验证完成: {{valid_count}}/{{len(data)}} 个样本有效")
    
    success_rate = valid_count / len(data) * 100 if data else 0
    print(f"📊 数据有效率: {{success_rate:.1f}}%")
    
    return success_rate > 90

if __name__ == "__main__":
    success = validate_data()
    if success:
        print("🎉 数据验证通过!")
    else:
        print("⚠️  数据验证失败，请检查数据质量")
'''
    
    script_path = f"{base_dir}/validate_data.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(validation_script)
    
    # 添加执行权限
    os.chmod(script_path, 0o755)
    
    print(f"  🔧 数据验证脚本: {script_path}")
    return script_path


def main():
    """主函数：创建完整的数据准备示例"""
    
    print("=" * 60)
    print("🎯 数据准备示例生成器")
    print("=" * 60)
    
    # 1. 获取示例数据结构
    example_data = create_real_world_example()
    
    # 2. 创建目录结构
    base_dir = prepare_data_structure("./example_data")
    
    # 3. 创建占位符图像和文本
    all_data = create_placeholder_images(example_data, base_dir)
    
    # 4. 创建JSON索引文件
    json_path = create_data_json(all_data, base_dir)
    
    # 5. 生成统计报告
    report_path = create_statistics_report(all_data, base_dir)
    
    # 6. 创建验证脚本
    validation_script = create_data_validation_script(base_dir)
    
    print("\n" + "=" * 60)
    print("✅ 数据准备示例创建完成!")
    print("=" * 60)
    
    print(f"\n📁 生成的文件:")
    print(f"  数据目录: {base_dir}/")
    print(f"  JSON索引: {json_path}")
    print(f"  统计报告: {report_path}")
    print(f"  验证脚本: {validation_script}")
    
    print(f"\n🚀 下一步:")
    print(f"  1. 查看示例数据: ls -la {base_dir}/")
    print(f"  2. 验证数据质量: python3 {validation_script}")
    print(f"  3. 替换为真实数据")
    print(f"  4. 开始训练: python3 train.py --data_dir {base_dir}")
    
    print(f"\n💡 使用提示:")
    print(f"  - 将占位符图像替换为您的真实图像")
    print(f"  - 修改文本描述以匹配真实图像内容")
    print(f"  - 保持相同的目录结构和JSON格式")
    print(f"  - 确保图像文件名与JSON中的路径一致")


if __name__ == "__main__":
    main()
