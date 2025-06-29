#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
练习4：用户数据向量化与降维可视化

目标：将用户画像数据转换为高维向量，并通过PCA降维可视化
核心技术：
1. BGE-m3模型计算文本嵌入
2. 特征向量拼接
3. PCA降维
4. 二维散点图可视化

作者：YjTech
版本：1.0
日期：2025年6月29日
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
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

class UserVectorizer:
    """用户数据向量化器"""
    
    def __init__(self, model_name='BAAI/bge-m3'):
        """
        初始化向量化器
        Args:
            model_name: BGE-m3模型名称
        """
        self.model_name = model_name
        self.embedding_model = None
        self.scaler = StandardScaler()
        self.pca = None
        
    def load_embedding_model(self):
        """加载BGE-m3嵌入模型"""
        print("正在加载BGE-m3嵌入模型...")
        try:
            # 尝试使用BGE-m3模型
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"✅ 成功加载模型: {self.model_name}")
        except Exception as e:
            print(f"⚠️  BGE-m3模型加载失败: {e}")
            print("使用备选模型: all-MiniLM-L6-v2")
            # 使用备选的轻量级模型
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为类别特征创建文本描述
        Args:
            df: 原始用户数据
        Returns:
            包含文本特征的DataFrame
        """
        print("创建文本特征描述...")
        
        # 为每个用户创建完整的文本描述
        text_features = []
        
        for _, row in df.iterrows():
            # 构建用户的自然语言描述
            user_text = f"用户性别{row['性别']}，居住在{row['所在城市']}，消费水平{row['消费水平']}，年龄{row['年龄']}岁，最近{row['最近活跃天数']}天内活跃"
            text_features.append(user_text)
        
        # 同时为每个类别特征单独创建描述
        categorical_texts = {
            '性别_文本': [f"性别{gender}" for gender in df['性别']],
            '城市_文本': [f"居住在{city}" for city in df['所在城市']],
            '消费_文本': [f"消费水平{level}" for level in df['消费水平']]
        }
        
        result_df = df.copy()
        result_df['用户描述'] = text_features
        
        for key, texts in categorical_texts.items():
            result_df[key] = texts
        
        print(f"创建了 {len(text_features)} 个用户文本描述")
        print(f"示例描述: {text_features[0]}")
        
        return result_df
    
    def compute_embeddings(self, df: pd.DataFrame) -> dict:
        """
        计算文本嵌入向量
        Args:
            df: 包含文本特征的DataFrame
        Returns:
            嵌入向量字典
        """
        print("计算文本嵌入向量...")
        
        if self.embedding_model is None:
            self.load_embedding_model()
        
        embeddings = {}
        text_columns = ['用户描述', '性别_文本', '城市_文本', '消费_文本']
        
        for col in text_columns:
            if col in df.columns:
                print(f"  处理 {col}...")
                texts = df[col].tolist()
                
                # 计算嵌入向量
                vectors = self.embedding_model.encode(texts, show_progress_bar=True)
                embeddings[col] = vectors
                
                print(f"  {col}: {vectors.shape[0]} 个文本 -> {vectors.shape[1]} 维向量")
        
        return embeddings
    
    def create_feature_vectors(self, df: pd.DataFrame, embeddings: dict) -> np.ndarray:
        """
        创建完整的特征向量
        Args:
            df: 原始数据
            embeddings: 嵌入向量字典
        Returns:
            特征向量矩阵
        """
        print("拼接特征向量...")
        
        feature_components = []
        
        # 1. 数值特征（标准化）
        numerical_features = ['年龄', '最近活跃天数']
        numerical_data = df[numerical_features].values
        numerical_data_scaled = self.scaler.fit_transform(numerical_data)
        feature_components.append(numerical_data_scaled)
        print(f"  数值特征: {numerical_data_scaled.shape[1]} 维")
        
        # 2. 嵌入向量特征
        for col, vectors in embeddings.items():
            feature_components.append(vectors)
            print(f"  {col}: {vectors.shape[1]} 维")
        
        # 3. 拼接所有特征
        final_vectors = np.hstack(feature_components)
        print(f"最终特征向量: {final_vectors.shape[0]} 个用户 × {final_vectors.shape[1]} 维特征")
        
        return final_vectors
    
    def apply_pca(self, vectors: np.ndarray, n_components: int = 2) -> tuple:
        """
        应用PCA降维
        Args:
            vectors: 高维特征向量
            n_components: 降维后的维度
        Returns:
            (降维后的向量, PCA对象)
        """
        print(f"应用PCA降维: {vectors.shape[1]} 维 -> {n_components} 维")
        
        self.pca = PCA(n_components=n_components, random_state=42)
        vectors_2d = self.pca.fit_transform(vectors)
        
        # 分析主成分
        explained_variance = self.pca.explained_variance_ratio_
        print(f"主成分解释的方差比例:")
        for i, var_ratio in enumerate(explained_variance):
            print(f"  PC{i+1}: {var_ratio:.1%}")
        print(f"  累计解释方差: {explained_variance.sum():.1%}")
        
        return vectors_2d, self.pca
    
    def visualize_2d_scatter(self, vectors_2d: np.ndarray, df: pd.DataFrame, save_plot: bool = True):
        """
        创建二维散点图可视化
        Args:
            vectors_2d: 二维降维向量
            df: 原始数据（用于着色）
            save_plot: 是否保存图表
        """
        print("创建二维散点图可视化...")
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('用户向量化PCA降维可视化', fontsize=16, fontweight='bold')
        
        # 1. 按性别着色
        ax1 = axes[0, 0]
        gender_colors = {'男': 'blue', '女': 'red', '未透露': 'gray'}
        for gender, color in gender_colors.items():
            mask = df['性别'] == gender
            if mask.any():
                ax1.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                           c=color, label=gender, alpha=0.6, s=30)
        ax1.set_xlabel('第一主成分 (PC1)')
        ax1.set_ylabel('第二主成分 (PC2)')
        ax1.set_title('按性别分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 按消费水平着色
        ax2 = axes[0, 1]
        consumption_colors = {'高': 'red', '中': 'orange', '低': 'green'}
        for level, color in consumption_colors.items():
            mask = df['消费水平'] == level
            if mask.any():
                ax2.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                           c=color, label=level, alpha=0.6, s=30)
        ax2.set_xlabel('第一主成分 (PC1)')
        ax2.set_ylabel('第二主成分 (PC2)')
        ax2.set_title('按消费水平分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 按年龄着色（连续色彩）
        ax3 = axes[1, 0]
        scatter = ax3.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                             c=df['年龄'], cmap='viridis', alpha=0.6, s=30)
        ax3.set_xlabel('第一主成分 (PC1)')
        ax3.set_ylabel('第二主成分 (PC2)')
        ax3.set_title('按年龄分布')
        plt.colorbar(scatter, ax=ax3, label='年龄')
        ax3.grid(True, alpha=0.3)
        
        # 4. 按城市着色
        ax4 = axes[1, 1]
        cities = df['所在城市'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(cities)))
        
        for i, city in enumerate(cities):
            mask = df['所在城市'] == city
            if mask.any():
                ax4.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                           c=[colors[i]], label=city, alpha=0.6, s=30)
        ax4.set_xlabel('第一主成分 (PC1)')
        ax4.set_ylabel('第二主成分 (PC2)')
        ax4.set_title('按城市分布')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('./output/user_vectorization_pca.png', dpi=300, bbox_inches='tight')
            print("可视化图表已保存为 './output/user_vectorization_pca.png'")
        
        plt.show()
    
    def analyze_clusters(self, vectors_2d: np.ndarray, df: pd.DataFrame):
        """
        分析用户聚类特征
        Args:
            vectors_2d: 二维降维向量
            df: 原始数据
        """
        print("\n" + "="*50)
        print("用户向量化聚类分析")
        print("="*50)
        
        # 计算不同特征在PC空间中的分布
        print("\n主成分统计:")
        print(f"PC1 范围: [{vectors_2d[:, 0].min():.2f}, {vectors_2d[:, 0].max():.2f}]")
        print(f"PC2 范围: [{vectors_2d[:, 1].min():.2f}, {vectors_2d[:, 1].max():.2f}]")
        
        # 分析不同类别在主成分空间的分布
        print(f"\n类别特征在主成分空间的中心点:")
        
        for feature in ['性别', '消费水平', '所在城市']:
            print(f"\n{feature}:")
            for category in df[feature].unique():
                mask = df[feature] == category
                if mask.any():
                    center_pc1 = vectors_2d[mask, 0].mean()
                    center_pc2 = vectors_2d[mask, 1].mean()
                    count = mask.sum()
                    print(f"  {category}: PC1={center_pc1:.2f}, PC2={center_pc2:.2f} ({count}个用户)")
        
        # 特征重要性分析
        if hasattr(self.pca, 'components_'):
            print(f"\n特征重要性分析 (前5个最重要的维度):")
            components = self.pca.components_
            
            for i in range(min(2, components.shape[0])):
                print(f"\nPC{i+1} 的主要贡献维度:")
                # 获取绝对值最大的5个特征
                feature_importance = np.abs(components[i])
                top_indices = np.argsort(feature_importance)[-5:][::-1]
                
                for idx in top_indices:
                    importance = components[i][idx]
                    print(f"  维度{idx}: {importance:.3f}")

def load_user_data():
    """加载用户数据"""
    try:
        df = pd.read_csv('./output/user_profiles.csv')
        print(f"成功加载用户数据: {len(df)} 条记录")
        return df
    except FileNotFoundError:
        print("未找到用户数据文件，请先运行 Exercise-1.py 生成数据")
        return None

def install_requirements():
    """安装必要的依赖包"""
    print("检查并安装必要的依赖包...")
    
    try:
        import sentence_transformers
        print("✅ sentence-transformers 已安装")
    except ImportError:
        print("⚠️  正在安装 sentence-transformers...")
        import subprocess
        subprocess.check_call(["pip", "install", "sentence-transformers"])
        print("✅ sentence-transformers 安装完成")

def main():
    """主函数"""
    print("练习4：用户数据向量化与降维可视化")
    print("="*50)
    
    # 检查依赖
    install_requirements()
    
    # 加载数据
    df = load_user_data()
    if df is None:
        return
    
    print(f"\n原始数据概览:")
    print(f"数据形状: {df.shape}")
    print(f"类别特征: {df.select_dtypes(include=['object']).columns.tolist()}")
    print(f"数值特征: {df.select_dtypes(include=['number']).columns.tolist()}")
    
    # 创建向量化器
    vectorizer = UserVectorizer()
    
    # 步骤1: 创建文本特征
    df_with_text = vectorizer.create_text_features(df)
    
    # 步骤2: 计算嵌入向量
    embeddings = vectorizer.compute_embeddings(df_with_text)
    
    # 步骤3: 创建特征向量
    feature_vectors = vectorizer.create_feature_vectors(df, embeddings)
    
    # 步骤4: PCA降维
    vectors_2d, pca = vectorizer.apply_pca(feature_vectors, n_components=2)
    
    # 步骤5: 可视化
    vectorizer.visualize_2d_scatter(vectors_2d, df, save_plot=True)
    
    # 步骤6: 聚类分析
    vectorizer.analyze_clusters(vectors_2d, df)
    
    # 保存结果
    result_df = df.copy()
    result_df['PC1'] = vectors_2d[:, 0]
    result_df['PC2'] = vectors_2d[:, 1]
    result_df.to_csv('./output/user_vectors_2d.csv', index=False, encoding='utf-8-sig')
    print(f"\n降维结果已保存到 './output/user_vectors_2d.csv'")
    
    print(f"\n向量化完成！")
    print(f"✅ 原始特征: {df.shape[1]} 维")
    print(f"✅ 高维向量: {feature_vectors.shape[1]} 维")
    print(f"✅ 降维向量: {vectors_2d.shape[1]} 维")
    print(f"✅ 方差解释: {pca.explained_variance_ratio_.sum():.1%}")
    
    return result_df, feature_vectors, vectors_2d

if __name__ == "__main__":
    # 运行主程序
    result_df, feature_vectors, vectors_2d = main()
