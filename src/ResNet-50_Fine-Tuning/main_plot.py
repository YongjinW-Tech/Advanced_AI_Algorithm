import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
import os
import random
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, processor, train=True):
        self.processor = processor
        self.images = []
        self.labels = []

        # 假设数据目录结构：data/train/positive/ 和 data/train/negative/
        positive_dir = os.path.join(data_dir, 'positive')
        negative_dir = os.path.join(data_dir, 'negative')
        
        # 加载正类图像
        if os.path.exists(positive_dir):
            for img_name in os.listdir(positive_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(positive_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(1)  # 正类标签
        
        # 加载负类图像
        if os.path.exists(negative_dir):
            for img_name in os.listdir(negative_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(negative_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(0)  # 负类标签
        
        # 打乱数据
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        self.images, self.labels = zip(*combined)
        
        # 划分训练集和测试集 (80:20)
        split_idx = int(0.8 * len(self.images))
        if train:
            self.images = self.images[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            self.images = self.images[split_idx:]
            self.labels = self.labels[split_idx:]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB') # convert()函数将图像转换为RGB模式

        # 使用processor处理图像
        inputs = self.processor(image, return_tensors="pt") # return_tensors="pt" # 返回PyTorch张量
        
        # 正确的调试方式：查看inputs的结构和pixel_values的形状
        # print(f"inputs keys: {inputs.keys()}")  # 查看inputs包含哪些键
        # print(f"pixel_values shape: {inputs['pixel_values'].shape}")  # 查看pixel_values的形状
        
        pixel_values = inputs['pixel_values'].squeeze(0)  # 移除batch维度
        # print(f"After squeeze - pixel_values shape: {pixel_values.shape}")  # 查看squeeze后的形状
        
        return pixel_values, torch.tensor(label, dtype=torch.long)

class ResNetBinaryClassifier(nn.Module):
    def __init__(self, pretrained_model_name="microsoft/resnet-50", freeze_backbone=True):
        super(ResNetBinaryClassifier, self).__init__()

        # 保存冻结状态，用于模型保存时的文件命名
        self.freeze_backbone = freeze_backbone
           
        # 加载预训练的ResNet模型
        # Ref: https://huggingface.co/microsoft/resnet-50
        self.resnet = ResNetForImageClassification.from_pretrained(pretrained_model_name)
        
        # 冻结主干网络参数（可选）freeze_backbone=True时，冻结主干网络参数
        if freeze_backbone:
            for param in self.resnet.resnet.parameters():
                param.requires_grad = False # param.requires_grad 的作用是控制参数是否参与梯度计算和优化
        
        # 打印分类器结构以便调试
        print("Original classifier structure:")
        print(self.resnet.classifier)
        # 输出：
        # Sequential(
        #     (0): Flatten(start_dim=1, end_dim=-1)
        #     (1): Linear(in_features=2048, out_features=1000, bias=True)
        # )
        
        # 从Sequential中提取Linear层的输入特征数
        # 上述ResNet50模型的分类器结构: Sequential(Flatten, Linear(2048, 1000))
        for module in self.resnet.classifier:
            if isinstance(module, nn.Linear):
                num_features = module.in_features # 取出Linear层的输入特征数
                break
        
        # 替换整个分类头为新的Sequential，保持相同结构但改为二分类
        self.resnet.classifier = nn.Sequential(  # 注意此处的结构要和原预训练模型中的分类头（分类器结构）保持一致
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(num_features, 2, bias=True)  # 二分类
        )
        
        print(f"Feature dimension: {num_features}") # 分类头前面的特征提取部分得到的特征维度，也就是分类头Linear层的输入特征数
        print(f"New classifier structure:")
        print(self.resnet.classifier)
    
    def forward(self, pixel_values):
        outputs = self.resnet(pixel_values=pixel_values)  # 使用pixel_values作为输入
        # outputs是一个包含logits的字典
        # logits是模型的输出，形状为(batch_size, num_classes)
        return outputs.logits

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_accuracy = 0.0

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': []
    }
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader): # batch_idx是当前批次的索引：[0, len(train_data)/batch_size)
            data, targets = data.to(device), targets.to(device)
            # 调试输出：打印当前batch的索引、数据和目标的形状
            # print(f"Batch {batch_idx}, Data shape: {data.shape}, Targets shape: {targets.shape}")  # 打印数据和目标的形状

            optimizer.zero_grad()               # 第1步：清零上一次的梯度
            outputs = model(data)               # 第2步：前向传播，计算预测
            loss = criterion(outputs, targets)  # 第3步：计算损失
            loss.backward()                     # 第4步：反向传播，计算梯度
            optimizer.step()                    # 第5步：根据梯度更新参数
            
            train_loss += loss.item()           # 累计训练损失（累加每个batch的损失值），其中 loss.item() 将PyTorch张量转换为Python数值
            _, predicted = torch.max(outputs.data, 1)   # 获取预测结果，torch.max() 返回最大值和对应的索引
            train_total += targets.size(0)      # 累计训练样本总数，targets.size(0) 获取当前batch的样本数量
            train_correct += (predicted == targets).sum().item()  # 计算预测正确的样本数
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predicted = []
        all_targets = []
        
        with torch.no_grad(): # 在验证阶段不需要计算梯度
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                all_predicted.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # 记录历史数据
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%')
        print('-' * 50)
        
        # 保存最佳模型 - 修改文件名以标记冻结状态
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            
            # 根据冻结状态生成不同的文件名
            freeze_status = "frozen" if model.freeze_backbone else "unfrozen"
            model_filename = f'./checkpoints/best_resnet_binary_classifier_{freeze_status}_backbone.pth'
            
            # 确保checkpoints目录存在
            os.makedirs('./checkpoints', exist_ok=True)
            
            torch.save(model.state_dict(), model_filename)
            print(f'New best model saved with accuracy: {best_accuracy:.2f}%')
            print(f'Model saved as: {model_filename}')
    
    # 打印最终分类报告
    print("\n最终验证集分类报告:")
    print(classification_report(all_targets, all_predicted, target_names=['Negative', 'Positive']))
    
    return model, history

def predict_images(trained_model, processor, test_dir, device):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.to(device)
    
    predict_results = []
    
    if os.path.exists(test_dir):
        # 收集测试图像
        test_images = []
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(test_dir, file))
        
        if test_images:
            print(f"Found {len(test_images)} test images")
            print("=" * 60)

            # Predict each image
            total_num_images = len(test_images)
            is_correct_count = 0
            for i, test_image in enumerate(test_images): 
                predicted_class, confidence = predict_single_image(trained_model, processor, test_image, device)
                
                # 根据图像名称判断真实类别（用于对比）
                filename = os.path.basename(test_image).lower() # 获取文件名并转换为小写
                if 'ultraman' in filename or 'Ultraman' in filename:
                    actual_class = "Ultraman (Positive)"
                    actual_label = 1
                else:
                    actual_class = "Other Class (Negative)"
                    actual_label = 0
                
                # 预测结果
                predicted_name = "Ultraman (Positive)" if predicted_class == 1 else "Other Class (Negative)"

                # 判断预测是否正确
                is_correct = True if predicted_class == actual_label else False
                is_correct_count += 1 if is_correct else 0
                
                # 保存预测结果到列表
                result_dict = {
                    "filename": os.path.basename(test_image),  # 原始文件名（不转小写）
                    "predicted_class": predicted_class,        # 预测类别数值 (0 or 1)
                    "predicted_name": predicted_name,          # 预测类别名称
                    "confidence": confidence,                  # 置信度
                    "actual_class": actual_class,              # 实际类别名称
                    "actual_label": actual_label,              # 实际类别数值
                    "is_correct": is_correct                   # 是否预测正确
                }
                predict_results.append(result_dict)
                
                # 打印结果
                status_symbol = "✓" if is_correct else "✗"
                print(f"Image {i+1}: {os.path.basename(test_image)}")
                print(f"  Actual Class: {actual_class}")
                print(f"  Predicted Class: {predicted_name}")
                print(f"  Confidence: {confidence:.4f}")
                print(f"  Prediction Result: {status_symbol}")
                print("-" * 40)
            print(f"\nTotal images: {total_num_images}, Correct predictions: {is_correct_count}, Accuracy: {is_correct_count / total_num_images:.2%}")
        else:
            print("Test directory is empty or no valid images found.")
    else:
        print(f"Test directory {test_dir} does not exist.")

    return predict_results

def predict_single_image(model, processor, image_path, device):
    """预测单张图像"""
    model.eval()
    
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence


def plot_training_comparison(frozen_history, unfrozen_history, save_path='./results/'):
    """
    可视化冻结和不冻结主干网络的训练对比
    
    Args:
        frozen_history: 冻结主干网络的训练历史
        unfrozen_history: 不冻结主干网络的训练历史
        save_path: 图片保存路径
    """
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 设置中文字体（如果需要显示中文）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ResNet-50 Fine-tuning: Frozen vs Unfrozen Backbone Comparison', fontsize=16, fontweight='bold')
    
    # 1. 训练损失对比
    axes[0, 0].plot(frozen_history['epochs'], frozen_history['train_loss'], 
                    'b-', label='Frozen Backbone', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(unfrozen_history['epochs'], unfrozen_history['train_loss'], 
                    'r-', label='Unfrozen Backbone', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 验证损失对比
    axes[0, 1].plot(frozen_history['epochs'], frozen_history['val_loss'], 
                    'b-', label='Frozen Backbone', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(unfrozen_history['epochs'], unfrozen_history['val_loss'], 
                    'r-', label='Unfrozen Backbone', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 训练准确率对比
    axes[1, 0].plot(frozen_history['epochs'], frozen_history['train_acc'], 
                    'b-', label='Frozen Backbone', linewidth=2, marker='o', markersize=4)
    axes[1, 0].plot(unfrozen_history['epochs'], unfrozen_history['train_acc'], 
                    'r-', label='Unfrozen Backbone', linewidth=2, marker='s', markersize=4)
    axes[1, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 验证准确率对比
    axes[1, 1].plot(frozen_history['epochs'], frozen_history['val_acc'], 
                    'b-', label='Frozen Backbone', linewidth=2, marker='o', markersize=4)
    axes[1, 1].plot(unfrozen_history['epochs'], unfrozen_history['val_acc'], 
                    'r-', label='Unfrozen Backbone', linewidth=2, marker='s', markersize=4)
    axes[1, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    comparison_path = os.path.join(save_path, 'training_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Training comparison plot saved to: {comparison_path}")
    
    plt.show()

def plot_single_training_history(history, title_suffix="", save_path='./results/'):
    """
    可视化单个模型的训练历史
    
    Args:
        history: 训练历史字典
        title_suffix: 标题后缀
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Training History {title_suffix}', fontsize=14, fontweight='bold')
    
    # 损失曲线
    axes[0].plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['epochs'], history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(history['epochs'], history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Accuracy Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"training_history_{title_suffix.lower().replace(' ', '_')}.png"
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_file}")
    
    plt.show()

def save_training_data(history, filename, save_path='./results/'):
    """保存训练数据到JSON文件"""
    import json
    
    os.makedirs(save_path, exist_ok=True)
    
    filepath = os.path.join(save_path, filename)
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training data saved to: {filepath}")

def run_comparison_experiment():
    """运行冻结和不冻结主干网络的对比实验"""
    
    # 数据目录
    train_data_dir = "./data/train"
    test_data_dir = "./data/test"
    
    # 初始化处理器
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    
    # 创建数据集
    print("Loading Data...")
    train_dataset = CustomImageDataset(train_data_dir, processor, train=True)
    val_dataset = CustomImageDataset(train_data_dir, processor, train=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 实验1: 冻结主干网络
    print("="*60)
    print("实验1: 冻结主干网络训练")
    print("="*60)
    
    model_frozen = ResNetBinaryClassifier(freeze_backbone=True)
    trained_model_frozen, frozen_history = train_model(
        model_frozen, train_loader, val_loader, num_epochs=20
    )
    
    # 保存冻结模型的训练历史
    save_training_data(frozen_history, 'frozen_backbone_history.json')
    
    # 可视化冻结模型的训练过程
    plot_single_training_history(frozen_history, "(Frozen Backbone)")
    
    print("\n" + "="*60)
    print("实验2: 不冻结主干网络训练")
    print("="*60)
    
    # 实验2: 不冻结主干网络
    model_unfrozen = ResNetBinaryClassifier(freeze_backbone=False)
    trained_model_unfrozen, unfrozen_history = train_model(
        model_unfrozen, train_loader, val_loader, num_epochs=20
    )
    
    # 保存不冻结模型的训练历史
    save_training_data(unfrozen_history, 'unfrozen_backbone_history.json')
    
    # 可视化不冻结模型的训练过程
    plot_single_training_history(unfrozen_history, "(Unfrozen Backbone)")
    
    # 对比可视化
    print("\n" + "="*60)
    print("生成对比可视化")
    print("="*60)
    
    plot_training_comparison(frozen_history, unfrozen_history)
    
    # 打印对比总结
    print_comparison_summary(frozen_history, unfrozen_history)
    
    # 在测试集上评估两个模型
    print("\n" + "="*60)
    print("测试集评估")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n冻结主干网络模型测试结果:")
    frozen_results = predict_images(trained_model_frozen, processor, test_data_dir, device)
    
    print("\n不冻结主干网络模型测试结果:")
    unfrozen_results = predict_images(trained_model_unfrozen, processor, test_data_dir, device)
    
    return {
        'frozen_model': trained_model_frozen,
        'unfrozen_model': trained_model_unfrozen,
        'frozen_history': frozen_history,
        'unfrozen_history': unfrozen_history,
        'frozen_test_results': frozen_results,
        'unfrozen_test_results': unfrozen_results
    }

def print_comparison_summary(frozen_history, unfrozen_history):
    """打印对比实验的总结"""
    print("\n" + "="*60)
    print("实验结果总结")
    print("="*60)
    
    # 最终性能对比
    frozen_final_acc = frozen_history['val_acc'][-1]
    unfrozen_final_acc = unfrozen_history['val_acc'][-1]
    
    frozen_best_acc = max(frozen_history['val_acc'])
    unfrozen_best_acc = max(unfrozen_history['val_acc'])
    
    frozen_final_loss = frozen_history['val_loss'][-1]
    unfrozen_final_loss = unfrozen_history['val_loss'][-1]
    
    print(f"冻结主干网络:")
    print(f"  最终验证准确率: {frozen_final_acc:.2f}%")
    print(f"  最佳验证准确率: {frozen_best_acc:.2f}%")
    print(f"  最终验证损失: {frozen_final_loss:.4f}")
    
    print(f"\n不冻结主干网络:")
    print(f"  最终验证准确率: {unfrozen_final_acc:.2f}%")
    print(f"  最佳验证准确率: {unfrozen_best_acc:.2f}%")
    print(f"  最终验证损失: {unfrozen_final_loss:.4f}")
    
    print(f"\n性能提升:")
    acc_improvement = unfrozen_best_acc - frozen_best_acc
    print(f"  准确率提升: {acc_improvement:+.2f}%")
    
    if acc_improvement > 0:
        print("  结论: 不冻结主干网络效果更好")
    elif acc_improvement < -1:
        print("  结论: 冻结主干网络效果更好")
    else:
        print("  结论: 两种方法效果相近")

def main():
    """主函数 - 可以选择运行单个实验或对比实验"""
    # 工作区目录为：/Users/wyjtech/learningspace/Advanced_AI_Algorithm/src/ResNet-50_Fine-Tuning

    # 选择运行模式
    run_comparison = True  # 设置为True运行对比实验，False运行单个实验
    
    if run_comparison:
        # 运行对比实验
        results = run_comparison_experiment()
        return results
    else: 
        # 数据目录，包含训练和测试数据
        train_data_dir = "./data/train" # 训练数据目录为 ./data/train
        test_data_dir = "./data/test" # 测试数据目录为 ./data/test

        # 初始化处理器
        # 负责 图像预处理（如调整尺寸、归一化、通道转换等），确保输入符合模型的要求。
        # 加载的是预处理配置文件（如 preprocessor_config.json）；不包含模型权重，仅包含数据处理逻辑。
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50") # 使用ResNet-50的数据处理器
        
        # 创建数据集
        print("Loading Data...")
        train_dataset = CustomImageDataset(train_data_dir, processor, train=True)
        val_dataset = CustomImageDataset(train_data_dir, processor, train=False)

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # 创建模型
        print("Initializing Model...")
        # freeze_backbone=True：冻结主干网络，只训练分类头
        # freeze_backbone=False：端到端微调整个网络
        model = ResNetBinaryClassifier(freeze_backbone=False)  # 冻结主干网络

        # 训练模型
        print("Training Model...")
        trained_model, history = train_model(model, train_loader, val_loader, num_epochs=20)
        
        # 可视化训练过程
        plot_single_training_history(history, "(Single Experiment)")
        
        # 预测测试图像
        print("\nPredicting Test Images:")
        predict_results = predict_images(trained_model, processor, test_data_dir, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # print("Prediction results:", predict_results)
        
        return {'model': trained_model, 'history': history, 'results': predict_results}

if __name__ == "__main__":
    main()


""" 预测类别提取
# outputs.data 的形状: [batch_size, num_classes]
# 例如 batch_size=8, num_classes=2:
# outputs = [[0.2, 0.8],   # 样本1: 类别0得分0.2, 类别1得分0.8 → 预测类别1
#           [0.9, 0.1],   # 样本2: 类别0得分0.9, 类别1得分0.1 → 预测类别0
#           [0.3, 0.7],   # 样本3: 类别0得分0.3, 类别1得分0.7 → 预测类别1
#           ...]

# torch.max(outputs.data, 1) 沿着第1维度(类别维度)找最大值
# 返回: (最大值, 最大值的索引)
values, predicted = torch.max(outputs.data, 1)
# values:    [0.8, 0.9, 0.7, ...]  # 最大得分
# predicted: [1,   0,   1,   ...]  # 对应的类别索引

# 通常只关心预测的类别，所以用 _ 忽略最大值
_, predicted = torch.max(outputs.data, 1)
"""