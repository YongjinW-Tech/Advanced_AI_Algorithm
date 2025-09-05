#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
练习4改进版：用户数据向量化与降维可视化优化

目标：优化降维效果，提高类别区分度
改进策略：
1. t-SNE和UMAP非线性降维
2. 标准化和特征工程
3. 多种可视化对比
4. 特征选择和权重调整

作者：YjTech
版本：2.0
日期：2025年6月29日
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Microsoft YaHei', 
    'Arial Unicode MS', 'Heiti TC', 'SimHei', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

class ImprovedUserVectorizer:
    """改进的用户数据向量化器"""
    
    def __init__(self, model_name='BAAI/bge-m3'):
        """
        初始化向量化器
        Args:
            model_name: BGE-m3模型名称
        """
        self.model_name = model_name
        self.embedding_model = None
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.feature_selector = None
        
    def load_embedding_model(self):
        """加载BGE-m3嵌入模型"""
        print("正在加载BGE-m3嵌入模型...")
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"✅ 成功加载模型: {self.model_name}")
        except Exception as e:
            print(f"⚠️  BGE-m3模型加载失败: {e}")
            print("使用备选模型: all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def create_enhanced_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建增强的文本特征
        Args:
            df: 原始用户数据
        Returns:
            包含增强文本特征的DataFrame
        """
        print("创建增强文本特征...")
        
        result_df = df.copy()
        
        # 1. 基础类别特征文本
        result_df['性别_文本'] = df['性别'].apply(lambda x: f"{x}性用户")
        result_df['城市_文本'] = df['所在城市'].apply(lambda x: f"来自{x}")
        result_df['消费_文本'] = df['消费水平'].apply(lambda x: f"{x}消费群体")
        
        # 2. 组合特征文本
        result_df['性别城市_文本'] = df.apply(lambda x: f"{x['性别']}性{x['所在城市']}用户", axis=1)
        result_df['消费年龄_文本'] = df.apply(lambda x: f"{x['消费水平']}消费{x['年龄']}岁用户", axis=1)
        
        # 3. 年龄分组文本
        def age_group(age):
            if age < 25:
                return "年轻用户"
            elif age < 35:
                return "青年用户"
            elif age < 45:
                return "中年用户"
            else:
                return "资深用户"
        
        result_df['年龄组_文本'] = df['年龄'].apply(age_group)
        
        # 4. 活跃度分组文本
        def activity_group(days):
            if days <= 7:
                return "高活跃用户"
            elif days <= 30:
                return "中活跃用户"
            else:
                return "低活跃用户"
        
        result_df['活跃度_文本'] = df['最近活跃天数'].apply(activity_group)
        
        # 5. 综合用户画像
        result_df['用户画像'] = df.apply(
            lambda x: f"{x['性别']}性{x['年龄']}岁{x['消费水平']}消费{x['所在城市']}用户，"
                     f"最近{x['最近活跃天数']}天活跃", axis=1
        )
        
        print(f"创建了 {len([col for col in result_df.columns if '_文本' in col or '画像' in col])} 个文本特征")
        
        return result_df
    
    def compute_embeddings(self, df: pd.DataFrame) -> dict:
        """计算文本嵌入向量"""
        print("计算文本嵌入向量...")
        
        if self.embedding_model is None:
            self.load_embedding_model()
        
        embeddings = {}
        text_columns = [col for col in df.columns if '_文本' in col or '画像' in col]
        
        for col in text_columns:
            print(f"  处理 {col}...")
            texts = df[col].tolist()
            vectors = self.embedding_model.encode(texts, show_progress_bar=True)
            embeddings[col] = vectors
            print(f"  {col}: {vectors.shape[0]} 个文本 -> {vectors.shape[1]} 维向量")
        
        return embeddings
    
    def create_enhanced_feature_vectors(self, df: pd.DataFrame, embeddings: dict) -> np.ndarray:
        """创建增强的特征向量"""
        print("拼接增强特征向量...")
        
        feature_components = []
        
        # 1. 数值特征（使用RobustScaler减少异常值影响）
        numerical_features = ['年龄', '最近活跃天数']
        numerical_data = df[numerical_features].values
        numerical_data_scaled = self.robust_scaler.fit_transform(numerical_data)
        
        # 增加数值特征的权重
        numerical_data_weighted = numerical_data_scaled * 3  # 增加权重
        feature_components.append(numerical_data_weighted)
        print(f"  数值特征: {numerical_data_weighted.shape[1]} 维 (加权)")
        
        # 2. 嵌入向量特征（标准化）
        for col, vectors in embeddings.items():
            # 对每个嵌入向量进行L2标准化
            vectors_normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            feature_components.append(vectors_normalized)
            print(f"  {col}: {vectors_normalized.shape[1]} 维 (L2标准化)")
        
        # 3. 拼接所有特征
        final_vectors = np.hstack(feature_components)
        print(f"最终特征向量: {final_vectors.shape[0]} 个用户 × {final_vectors.shape[1]} 维特征")
        
        return final_vectors
    
    def apply_feature_selection(self, vectors: np.ndarray, df: pd.DataFrame, 
                              target_feature: str = '消费水平', k: int = 1000) -> np.ndarray:
        """应用特征选择"""
        print(f"应用特征选择: {vectors.shape[1]} 维 -> {k} 维")
        
        # 使用消费水平作为目标进行特征选择
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df[target_feature])
        
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, vectors.shape[1]))
        vectors_selected = self.feature_selector.fit_transform(vectors, y)
        
        print(f"特征选择完成: 保留了 {vectors_selected.shape[1]} 个最重要的特征")
        return vectors_selected
    
    def apply_multiple_dimensionality_reduction(self, vectors: np.ndarray) -> dict:
        """应用多种降维方法"""
        print("应用多种降维方法...")
        
        results = {}
        
        # 1. PCA降维
        print("  执行PCA降维...")
        pca = PCA(n_components=2, random_state=42)
        vectors_pca = pca.fit_transform(vectors)
        results['PCA'] = {
            'vectors': vectors_pca,
            'model': pca,
            'explained_variance': pca.explained_variance_ratio_.sum()
        }
        print(f"    PCA解释方差: {pca.explained_variance_ratio_.sum():.1%}")
        
        # 2. t-SNE降维
        print("  执行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
                   learning_rate=200, max_iter=1000, verbose=0)
        vectors_tsne = tsne.fit_transform(vectors)
        results['t-SNE'] = {
            'vectors': vectors_tsne,
            'model': tsne
        }
        print("    t-SNE降维完成")
        
        # 3. 尝试安装和使用UMAP
        try:
            import umap
            print("  执行UMAP降维...")
            umap_model = umap.UMAP(n_components=2, random_state=42, 
                                  n_neighbors=15, min_dist=0.1)
            vectors_umap = umap_model.fit_transform(vectors)
            results['UMAP'] = {
                'vectors': vectors_umap,
                'model': umap_model
            }
            print("    UMAP降维完成")
        except ImportError:
            print("    UMAP未安装，跳过UMAP降维")
        
        return results
    
    def visualize_comparison(self, reduction_results: dict, df: pd.DataFrame, save_plot: bool = True):
        """对比可视化不同降维方法"""
        print("创建降维方法对比可视化...")
        
        n_methods = len(reduction_results)
        fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 10))
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('不同降维方法效果对比', fontsize=16, fontweight='bold')
        
        # 颜色设置
        gender_colors = {'男': 'blue', '女': 'red', '未透露': 'gray'}
        consumption_colors = {'高': 'red', '中': 'orange', '低': 'green'}
        
        for i, (method_name, result) in enumerate(reduction_results.items()):
            vectors_2d = result['vectors']
            
            # 第一行：按性别着色
            ax1 = axes[0, i]
            for gender, color in gender_colors.items():
                mask = df['性别'] == gender
                if mask.any():
                    ax1.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                               c=color, label=gender, alpha=0.6, s=30)
            ax1.set_title(f'{method_name} - 按性别分布')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 第二行：按消费水平着色
            ax2 = axes[1, i]
            for level, color in consumption_colors.items():
                mask = df['消费水平'] == level
                if mask.any():
                    ax2.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                               c=color, label=level, alpha=0.6, s=30)
            ax2.set_title(f'{method_name} - 按消费水平分布')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 添加方差解释信息（如果有）
            if 'explained_variance' in result:
                ax1.text(0.02, 0.98, f'解释方差: {result["explained_variance"]:.1%}', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('./output/dimensionality_reduction_comparison.png', dpi=300, bbox_inches='tight')
            print("对比图表已保存为 './output/dimensionality_reduction_comparison.png'")
        
        plt.show()
    
    def calculate_separation_metrics(self, vectors_2d: np.ndarray, df: pd.DataFrame, 
                                   feature: str = '消费水平') -> dict:
        """计算类别分离度指标"""
        print(f"计算 {feature} 的类别分离度...")
        
        categories = df[feature].unique()
        metrics = {}
        
        # 计算类内距离和类间距离
        intra_distances = []
        inter_distances = []
        
        for cat in categories:
            mask = df[feature] == cat
            if mask.sum() > 1:
                cat_points = vectors_2d[mask]
                cat_center = cat_points.mean(axis=0)
                
                # 类内距离（到中心的平均距离）
                intra_dist = np.mean(np.linalg.norm(cat_points - cat_center, axis=1))
                intra_distances.append(intra_dist)
                
                # 与其他类别中心的距离
                for other_cat in categories:
                    if other_cat != cat:
                        other_mask = df[feature] == other_cat
                        if other_mask.sum() > 0:
                            other_center = vectors_2d[other_mask].mean(axis=0)
                            inter_dist = np.linalg.norm(cat_center - other_center)
                            inter_distances.append(inter_dist)
        
        avg_intra = np.mean(intra_distances) if intra_distances else 0
        avg_inter = np.mean(inter_distances) if inter_distances else 0
        
        # 分离度指标：类间距离/类内距离
        separation_ratio = avg_inter / avg_intra if avg_intra > 0 else 0
        
        metrics = {
            'avg_intra_distance': avg_intra,
            'avg_inter_distance': avg_inter,
            'separation_ratio': separation_ratio
        }
        
        print(f"  平均类内距离: {avg_intra:.3f}")
        print(f"  平均类间距离: {avg_inter:.3f}")
        print(f"  分离度指标: {separation_ratio:.3f}")
        
        return metrics

def install_advanced_requirements():
    """安装高级依赖包"""
    print("检查并安装高级依赖包...")
    
    packages_to_check = [
        ('sentence_transformers', 'sentence-transformers'),
        ('umap', 'umap-learn')
    ]
    
    for package_name, pip_name in packages_to_check:
        try:
            __import__(package_name)
            print(f"✅ {pip_name} 已安装")
        except ImportError:
            print(f"⚠️  正在安装 {pip_name}...")
            import subprocess
            subprocess.check_call(["pip", "install", pip_name])
            print(f"✅ {pip_name} 安装完成")

def main():
    """主函数"""
    print("练习4改进版：用户数据向量化与降维可视化优化")
    print("="*60)
    
    # 检查依赖
    install_advanced_requirements()
    
    # 加载数据
    try:
        df = pd.read_csv('./output/user_profiles.csv')
        print(f"成功加载用户数据: {len(df)} 条记录")
    except FileNotFoundError:
        print("未找到用户数据文件，请先运行 Exercise-1_1.py 生成数据")
        return
    
    # 创建改进的向量化器
    vectorizer = ImprovedUserVectorizer()
    
    # 步骤1: 创建增强文本特征
    df_with_text = vectorizer.create_enhanced_text_features(df)
    
    # 步骤2: 计算嵌入向量
    embeddings = vectorizer.compute_embeddings(df_with_text)
    
    # 步骤3: 创建增强特征向量
    feature_vectors = vectorizer.create_enhanced_feature_vectors(df, embeddings)
    
    # 步骤4: 特征选择（可选）
    if feature_vectors.shape[1] > 1000:
        feature_vectors_selected = vectorizer.apply_feature_selection(
            feature_vectors, df, target_feature='消费水平', k=1000
        )
    else:
        feature_vectors_selected = feature_vectors
    
    # 步骤5: 应用多种降维方法
    reduction_results = vectorizer.apply_multiple_dimensionality_reduction(feature_vectors_selected)
    
    # 步骤6: 对比可视化
    vectorizer.visualize_comparison(reduction_results, df, save_plot=True)
    
    # 步骤7: 计算分离度指标
    print(f"\n{'='*50}")
    print("降维方法效果评估")
    print(f"{'='*50}")
    
    for method_name, result in reduction_results.items():
        print(f"\n{method_name} 方法:")
        metrics = vectorizer.calculate_separation_metrics(
            result['vectors'], df, feature='消费水平'
        )
        
        # 保存结果
        result_df = df.copy()
        result_df[f'{method_name}_X'] = result['vectors'][:, 0]
        result_df[f'{method_name}_Y'] = result['vectors'][:, 1]
        
        output_file = f'./output/user_vectors_{method_name.lower()}.csv'
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"  结果已保存到: {output_file}")
    
    # 推荐最佳方法
    print(f"\n{'='*50}")
    print("改进建议")
    print(f"{'='*50}")
    
    print("""
    基于分析结果，以下是改进降维效果的建议：
    
    1. 📊 数据层面改进：
       - 增加更多有区分性的特征
       - 处理类别不平衡问题
       - 收集更多样本数据
    
    2. 🔧 技术层面改进：
       - 尝试不同的文本构造策略
       - 调整特征权重和组合方式
       - 使用集成降维方法
    
    3. 📈 可视化改进：
       - 尝试3D可视化
       - 使用密度图显示聚类
       - 添加决策边界可视化
    
    4. 🎯 评估改进：
       - 使用聚类质量指标
       - 计算分类准确率
       - 进行交叉验证
    """)
    
    return reduction_results

if __name__ == "__main__":
    results = main()
