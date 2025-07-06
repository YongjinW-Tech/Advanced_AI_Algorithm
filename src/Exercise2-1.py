#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
练习2-1：基于语义理解的文本分类器

任务：情感分析 - 判断文本是"积极的"还是"消极的"

1. BGE-M3模型进行文本语义向量化
2. 逻辑回归进行二元分类
3. 手工构建的高质量样本数据集

作者：YjTech
版本：1.0
日期：2025年7月6日
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Microsoft YaHei', 
    'Arial Unicode MS', 'Heiti TC', 'SimHei', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

class SemanticTextClassifier:
    """基于语义理解的文本分类器"""
    
    def __init__(self, embedding_model_name='BAAI/bge-m3'):
        """
        初始化分类器
        Args:
            embedding_model_name: BGE-M3模型名称
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.classifier_name = 'LogisticRegression'
        self.classifier = LogisticRegression(random_state=42, max_iter=1000) # max_iter是最大迭代次数
        self.scaler = StandardScaler()  # 特征标准化器
        self.is_trained = False
        
    def load_embedding_model(self):
        """加载文本嵌入模型"""
        print(f"正在加载{self.embedding_model_name}嵌入模型...")
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"成功加载模型: {self.embedding_model_name}")
        except Exception as e:
            print(f"文本嵌入模型加载失败: {e}")
            print("使用备选模型: all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def prepare_sample_data(self):
        """准备高质量的情感分析样本数据"""
        # 积极情感样本 (标签: 1)
        positive_samples = [
            "这个产品的质量真的超出了我的期望，非常满意！",
            "今天的天气真好，阳光明媚，心情特别舒畅。",
            "感谢团队的辛勤付出，这次项目取得了圆满成功。",
            "孩子们的笑声是世界上最美妙的音乐。",
            "这家餐厅的服务态度很棒，菜品也很美味。",
            "经过努力学习，终于通过了考试，太开心了！",
            "新的工作环境很不错，同事们都很友善。",
            "这本书写得非常精彩，强烈推荐给大家。",
            "今天收到了意外的好消息，整个人都充满了正能量。",
            "和朋友们一起度过的时光总是那么愉快和难忘。",
            "这次旅行的经历让我收获满满，见识了很多美景。",
            "客服的专业解答帮我解决了问题，服务很到位。",
            "看到自己的进步，感觉所有的努力都是值得的。",
            "这个创意真的很棒，让人眼前一亮。"
        ]
        
        # 消极情感样本 (标签: 0)
        negative_samples = [
            "这个产品的质量实在太差了，完全不值这个价格。",
            "今天遇到了很多烦心事，心情糟糕透了。",
            "对这次的服务感到非常失望，完全没有达到预期。",
            "排队等了这么久，结果还是没有解决问题。",
            "这家店的态度太恶劣了，以后再也不会来了。",
            "工作压力太大了，感觉快要承受不住了。",
            "这次的体验真的很糟糕，浪费了时间和金钱。",
            "产品存在明显的质量问题，希望能够改进。",
            "客服的回复很敷衍，根本没有解决实际问题。",
            "这个政策的实施给我们带来了很多不便。",
            "连续加班让我身心疲惫，工作效率也下降了。",
            "这次购物的经历让我对这个品牌失去了信心。",
            "系统经常出故障，严重影响了正常使用。",
            "价格虚高，性价比很低，不推荐购买。"
        ]
        
        # 构建数据集
        texts = positive_samples + negative_samples 
        labels = [1] * len(positive_samples) + [0] * len(negative_samples)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'text': texts,
            'label': labels,
            'sentiment': ['积极' if label == 1 else '消极' for label in labels]
        })
        
        print(f"样本数据准备完成: 积极样本 {len(positive_samples)} 条，消极样本 {len(negative_samples)} 条，总计 {len(texts)} 条")
        
        return df
    
    def extract_features(self, texts):
        """
        使用BGE-M3模型提取文本的语义特征
        Args:
            texts: 文本列表
        Returns:
            语义向量矩阵
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        # 将文本转换为语义向量
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,     # 显示进度条
            normalize_embeddings=True   # 归一化嵌入向量
        )
        
        print(f"样本特征提取完成: {embeddings.shape[0]} 个样本 × {embeddings.shape[1]} 维特征")
        print(f"第一个样本的特征向量前 5 个维度: {embeddings[0][:5]}...")  # 打印前5个维度
        return embeddings
    
    def train_classifier(self, X_train, y_train):
        """
        训练逻辑回归分类器
        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        print(f"正在训练{self.classifier_name}分类器...")
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train) # 标准化训练特征
        # 训练分类器
        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True
        # 计算训练准确率
        train_accuracy = self.classifier.score(X_train_scaled, y_train)
        print(f"模型训练完成，训练准确率: {train_accuracy:.3f}")
        
        return train_accuracy
    
    def evaluate_model(self, X_test, y_test):
        """
        评估模型性能
        Args:
            X_test: 测试特征
            y_test: 测试标签
        Returns:
            评估指标字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train_classifier 方法")
        
        print("正在评估模型性能...")
        # 标准化测试特征
        X_test_scaled = self.scaler.transform(X_test)
        # 预测
        y_pred = self.classifier.predict(X_test_scaled)
        y_prob = self.classifier.predict_proba(X_test_scaled)
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"模型评估完成，测试准确率: {accuracy:.3f}")
        
        # 详细分类报告
        print("\n详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=['消极', '积极']))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_prob
        }
    
    def predict_new_texts(self, texts):
        """
        预测新文本的情感
        Args:
            texts: 新文本列表
        Returns:
            预测结果字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train_classifier 方法")
        
        print("正在预测新文本的情感...")
        # 提取特征
        embeddings = self.extract_features(texts)
        # 标准化特征
        embeddings_scaled = self.scaler.transform(embeddings)
        # 预测
        predictions = self.classifier.predict(embeddings_scaled)
        probabilities = self.classifier.predict_proba(embeddings_scaled)
        
        # 整理结果
        results = []
        for i, text in enumerate(texts):
            sentiment = '积极' if predictions[i] == 1 else '消极'
            confidence = max(probabilities[i])  # 置信度为最大概率
            results.append({
                'text': text,
                'prediction': predictions[i],
                'sentiment': sentiment,
                'confidence': confidence,
                'positive_prob': probabilities[i][1],
                'negative_prob': probabilities[i][0]
            })
        
        return results
    
    def visualize_results(self, df, X, y, save_plot=True):
        """
        可视化分析结果 - 3D PCA降维可视化
        Args:
            df: 原始数据
            X: 特征向量
            y: 标签
            save_plot: 是否保存图表
        """
        print("创建3D PCA降维可视化图表...")
        
        # 导入3D绘图库
        from sklearn.decomposition import PCA
        from mpl_toolkits.mplot3d import Axes3D
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 8))  # 创建图形对象，fig 是一个 matplotlib 的 Figure 对象
        # 在一个图形对象中添加一个3D子图
        ax = fig.add_subplot(111, projection='3d') # add_subplot() 是 Figure 对象的方法，用于添加子图，不指定projection（默认2D）;此处返回一个 Axes 对象，这里是3D的Axes3D对象
        # 111 是子图网格的简化表示 --> 格式：nrows(行数) + ncols(列数) + index(位置索引)
        # 111 = 1 + 1 + 1
        # - 第一个1：总共1行
        # - 第二个1：总共1列  
        # - 第三个1：这个子图位于第1个位置
        
        # PCA降维到3维
        pca = PCA(n_components=3, random_state=42)
        X_pca = pca.fit_transform(X)    # fit_transform() 方法同时进行拟合和转换，返回降维后的数据
        
        # 按情感分类着色
        colors = ['red', 'green']
        labels = ['消极', '积极']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = y == i  # 创建布尔掩码，选择对应情感的样本
            # 假设 y 是标签数组 y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, ...])  # 28个样本的标签
            # 第1次循环：i=0 (消极类别)  mask = y == 0
            # - 标记出所有消极样本的位置：mask = [True, False, True, False, False, True, False, True, True, False, ...]
            # 第2次循环：i=1 (积极类别)  mask = y == 1
            # - 标记出所有积极样本的位置：mask = [False, True, False, True, True, False, True, False, False, True, ...]
            ax.scatter(X_pca[mask, 0],   # X_pca[mask, 0] 是第1个主成分（作为 x 轴）
                       X_pca[mask, 1],   # X_pca[mask, 1] 是第2个主成分（作为 y 轴）
                       X_pca[mask, 2],   # X_pca[mask, 2] 是第3个主成分（作为 z 轴）
                       c=color, label=label, 
                       alpha=0.7,        # 透明度
                       s=60              # 点的大小
                       )
            # ax.scatter(x, y, z) 方法用于在3D空间中绘制散点图
            # X_pca[mask, 0]  # 布尔索引 + 列索引的组合
            #         ↑      ↑
            #         |      └─ 第0列（PC1坐标）
            #         └─ 布尔掩码（选择特定类别的行）
            # 等价于：
            # selected_rows = X_pca[mask]     # 先选择行
            # pc1_coords = selected_rows[:, 0] # 再选择列
        
        # 设置图表标题和标签
        ax.set_title(f'BGE-M3特征向量3D PCA可视化\n解释方差: {pca.explained_variance_ratio_.sum():.1%}', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=12)
        
        # 添加图例
        ax.legend(loc='upper right', fontsize=12)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 打印PCA信息
        print(f"PCA降维信息:")
        print(f"  - 原始特征维度: {X.shape[1]}")    # X.shape[0] 是样本数量，X.shape[1] 是特征维度
        print(f"  - 降维后维度: {X_pca.shape[1]}")
        print(f"  - 解释方差比例: {pca.explained_variance_ratio_}")
        print(f"  - 累计解释方差: {pca.explained_variance_ratio_.sum():.1%}")
        
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        
        if save_plot:
            output_path = './output/Exer2-1_sentiment_analysis_results.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"3D PCA可视化图表已保存: {output_path}")
        
        plt.show()

def main():
    """主函数 - 完整的情感分析流程"""
    # 0. 初始化分类器
    classifier = SemanticTextClassifier()
    # 1. 准备样本数据
    print("Step 1: 准备情感分析样本数据...")
    df = classifier.prepare_sample_data()
    # 2. 提取语义特征
    print("Step 2: 正在提取文本语义特征...")
    embeddings = classifier.extract_features(df['text'].tolist())
    # 3. 数据集划分
    print("Step 3: 正在划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, df['label'].values, 
        test_size=0.3,          # 测试集占比30%
        random_state=42,        # 保证可重复性
        stratify=df['label']    # 分层抽样，保持标签比例
    )
    print(f"数据划分完成: 训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    # 4. 训练分类器
    print("Step 4: 模型训练")
    train_accuracy = classifier.train_classifier(X_train, y_train)
    # 5. 评估模型
    print("Step 5: 模型评估")
    eval_results = classifier.evaluate_model(X_test, y_test)
    # 6. 测试新文本
    print("Step 6: 新文本预测测试")
    # 准备一些新的测试文本
    new_texts = [
        "这个电影真的太精彩了，我看了好几遍都不腻！",
        "服务态度很差，让人感到很不愉快。",
        "今天的会议很有收获，学到了很多新知识。",
        "这次购物体验让我很失望，产品质量不如预期。",
        "和家人一起度过的周末总是那么温馨美好。",
        "网络连接不稳定，严重影响了工作效率。"
    ]
    prediction_results = classifier.predict_new_texts(new_texts)
    for i, result in enumerate(prediction_results, 1): # enumerate(iterable, start=0),此处 start=1，所以索引计数器 i 从1开始
        print(f"{i}. 文本: {result['text']}")
        print(f"   预测情感: {result['sentiment']} (置信度: {result['confidence']:.3f})")
        print(f"   详细概率: 积极={result['positive_prob']:.3f}, 消极={result['negative_prob']:.3f}")
        print()
    # 7. 可视化结果
    print("Step 7: 结果可视化")
    classifier.visualize_results(df, embeddings, df['label'].values)
    
    return classifier, df, embeddings, prediction_results

if __name__ == "__main__":
    # 运行完整的情感分析流程
    classifier, data, features, results = main()
