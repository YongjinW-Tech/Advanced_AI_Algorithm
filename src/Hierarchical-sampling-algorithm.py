#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层抽样算法实现 (Stratified Sampling Algorithm)

目标：手动实现分层抽样算法，模拟C/C++实现思路
应用：机器学习面试高频题，数据预处理核心算法

算法思路：
1. 按目标变量分组（分层）
2. 在每层内部进行随机抽样
3. 保持各层样本比例与原始数据一致
4. 合并各层抽样结果

作者：YjTech
版本：1.0
日期：2025年6月29日
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class StratifiedSampler:
    """
    分层抽样器类
    模拟C/C++实现思路，使用面向对象设计
    """
    
    def __init__(self, random_seed: int = 42):
        """
        初始化分层抽样器
        Args:
            random_seed: 随机种子
        """
        self.random_seed = random_seed
        self.reset_random_state()
        
    def reset_random_state(self):
        """重置随机状态"""
        np.random.seed(self.random_seed)    # 设置NumPy随机种子
        random.seed(self.random_seed)       # 设置Python标准库的随机种子
    
    def stratified_split(self, 
                        X: np.ndarray, 
                        y: np.ndarray, 
                        test_size: float = 0.2,
                        shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        分层抽样主算法
        
        Args:
            X: 特征数据 (n_samples, n_features)
            y: 目标变量 (n_samples,)
            test_size: 测试集比例
            shuffle: 是否随机打乱
            
        Returns:
            X_train, X_test, y_train, y_test
            
        算法复杂度：
            时间复杂度: O(n log n) - 主要是排序操作
            空间复杂度: O(n) - 存储索引和分组信息
        """
        print(f"开始分层抽样算法...")
        print(f"数据规模: {len(X)} 样本, {X.shape[1] if len(X.shape) > 1 else 1} 特征")
        print(f"测试集比例: {test_size:.1%}")
        
        # 步骤1: 数据验证
        if len(X) != len(y):
            raise ValueError(f"特征和标签数量不匹配: {len(X)} vs {len(y)}")
        
        if not 0 < test_size < 1:
            raise ValueError(f"test_size必须在(0,1)之间: {test_size}")
        
        # 步骤2: 构建分层索引字典 - O(n)
        print("\n步骤1: 构建分层索引...")
        strata_indices = self._build_strata_indices(y)
        
        # 步骤3: 计算每层的抽样数量 - O(k), k为类别数
        print("步骤2: 计算抽样数量...")
        sampling_plan = self._calculate_sampling_sizes(strata_indices, test_size)
        
        # 步骤4: 每层内部抽样 - O(n)
        print("步骤3: 执行分层抽样...")
        train_indices, test_indices = self._perform_stratified_sampling(
            strata_indices, sampling_plan, shuffle
        )
        
        # 步骤5: 构造结果数据集 - O(n)
        print("步骤4: 构造结果数据集...")
        X_train, X_test, y_train, y_test = self._build_result_datasets(
            X, y, train_indices, test_indices, shuffle
        )
        
        # 步骤6: 验证结果
        self._validate_results(y, y_train, y_test, test_size)
        
        print(f"分层抽样完成!")
        return X_train, X_test, y_train, y_test
    
    def _build_strata_indices(self, y: np.ndarray) -> Dict[Any, List[int]]:
        """
        构建分层索引字典
        
        Args:
            y: 目标变量
            
        Returns:
            strata_indices: {类别: [索引列表]}
            
        时间复杂度: O(n)
        """
        strata_indices = defaultdict(list)
        
        # 遍历所有样本，按类别分组
        for idx, label in enumerate(y):
            strata_indices[label].append(idx)
        
        # 打印分层信息
        print(f"  发现 {len(strata_indices)} 个分层:")
        for label, indices in strata_indices.items():
            print(f"    {label}: {len(indices)} 样本 ({len(indices)/len(y):.1%})")
        
        return dict(strata_indices)
    
    def _calculate_sampling_sizes(self, 
                                 strata_indices: Dict[Any, List[int]], 
                                 test_size: float) -> Dict[Any, Tuple[int, int]]:
        """
        计算每层的训练集和测试集抽样数量
        
        Args:
            strata_indices: 分层索引字典
            test_size: 测试集比例
            
        Returns:
            sampling_plan: {类别: (训练集数量, 测试集数量)}
            
        时间复杂度: O(k), k为类别数
        """
        sampling_plan = {}
        total_samples = sum(len(indices) for indices in strata_indices.values())
        
        print(f"  抽样计划:")
        for label, indices in strata_indices.items():
            stratum_size = len(indices)
            
            # 计算该层的测试集数量 - 四舍五入确保整数
            test_count = round(stratum_size * test_size)
            train_count = stratum_size - test_count
            
            # 确保每层至少有一个样本（如果原始层有样本的话）
            if stratum_size > 0:
                test_count = max(1, min(test_count, stratum_size - 1)) if stratum_size > 1 else 0
                train_count = stratum_size - test_count
            
            sampling_plan[label] = (train_count, test_count)
            print(f"    {label}: 训练集{train_count}, 测试集{test_count}")
        
        return sampling_plan
    
    def _perform_stratified_sampling(self, 
                                   strata_indices: Dict[Any, List[int]], 
                                   sampling_plan: Dict[Any, Tuple[int, int]],
                                   shuffle: bool) -> Tuple[List[int], List[int]]:
        """
        执行分层抽样
        
        Args:
            strata_indices: 分层索引字典
            sampling_plan: 抽样计划
            shuffle: 是否随机打乱
            
        Returns:
            train_indices, test_indices: 训练集和测试集索引列表
            
        时间复杂度: O(n log n) - 主要是随机打乱操作
        """
        train_indices = []
        test_indices = []
        
        for label, indices in strata_indices.items():
            train_count, test_count = sampling_plan[label]
            
            # 复制索引列表，避免修改原数据
            stratum_indices = indices.copy()
            
            # 层内随机打乱 - O(m log m), m为该层样本数
            if shuffle:
                random.shuffle(stratum_indices)
            
            # 按计划抽样
            stratum_train = stratum_indices[:train_count]
            stratum_test = stratum_indices[train_count:train_count + test_count]
            
            # 累积到全局索引列表
            train_indices.extend(stratum_train)
            test_indices.extend(stratum_test)
            
            print(f"    {label}: 实际抽取训练集{len(stratum_train)}, 测试集{len(stratum_test)}")
        
        return train_indices, test_indices
    
    def _build_result_datasets(self, 
                              X: np.ndarray, 
                              y: np.ndarray, 
                              train_indices: List[int], 
                              test_indices: List[int],
                              shuffle: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构造最终的数据集
        
        Args:
            X, y: 原始数据
            train_indices, test_indices: 训练集和测试集索引
            shuffle: 是否最终打乱
            
        Returns:
            X_train, X_test, y_train, y_test
            
        时间复杂度: O(n)
        """
        # 按索引提取数据 - O(n)
        X_train = X[train_indices]
        X_test = X[test_indices] 
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # 最终随机打乱（可选）
        if shuffle:
            # 训练集打乱
            train_perm = np.random.permutation(len(train_indices))
            X_train = X_train[train_perm]
            y_train = y_train[train_perm]
            
            # 测试集打乱
            test_perm = np.random.permutation(len(test_indices))
            X_test = X_test[test_perm]
            y_test = y_test[test_perm]
        
        print(f"  最终数据集: 训练集{len(X_train)}, 测试集{len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def _validate_results(self, 
                         y_original: np.ndarray, 
                         y_train: np.ndarray, 
                         y_test: np.ndarray, 
                         expected_test_size: float):
        """
        验证分层抽样结果
        
        Args:
            y_original: 原始标签
            y_train, y_test: 训练集和测试集标签
            expected_test_size: 期望的测试集比例
        """
        print(f"\n结果验证:")
        
        # 验证数据完整性
        total_samples = len(y_train) + len(y_test)
        assert total_samples == len(y_original), f"样本数量不匹配: {total_samples} vs {len(y_original)}"
        
        # 验证测试集比例
        actual_test_size = len(y_test) / len(y_original)
        print(f"  测试集比例: 期望{expected_test_size:.1%}, 实际{actual_test_size:.1%}")
        
        # 验证分布一致性
        original_dist = self._calculate_distribution(y_original)
        train_dist = self._calculate_distribution(y_train)
        test_dist = self._calculate_distribution(y_test)
        
        print(f"  分布一致性检验:")
        max_deviation = 0
        for label in original_dist:
            orig_prop = original_dist[label]
            train_prop = train_dist.get(label, 0)
            test_prop = test_dist.get(label, 0)
            
            train_dev = abs(train_prop - orig_prop)
            test_dev = abs(test_prop - orig_prop)
            max_deviation = max(max_deviation, train_dev, test_dev)
            
            print(f"    {label}: 原始{orig_prop:.1%}, 训练{train_prop:.1%}(偏差{train_dev:+.1%}), 测试{test_prop:.1%}(偏差{test_dev:+.1%})")
        
        print(f"  最大偏差: {max_deviation:.1%}")
        
        if max_deviation < 0.02:
            print(f"  ✅ 分层抽样效果优秀!")
        elif max_deviation < 0.05:
            print(f"  ✅ 分层抽样效果良好!")
        else:
            print(f"  ⚠️  分层抽样效果一般，可能需要更大数据集")
    
    def _calculate_distribution(self, y: np.ndarray) -> Dict[Any, float]:
        """计算标签分布"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {label: count / total for label, count in zip(unique, counts)}

def compare_with_sklearn():
    """
    与sklearn的train_test_split对比验证
    """
    print("\n" + "="*60)
    print("与sklearn实现对比验证")
    print("="*60)
    
    # 生成测试数据
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, 
                              n_clusters_per_class=1, random_state=42)
    
    print(f"测试数据: {len(X)} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")
    
    # 我们的实现
    print(f"\n【我们的实现】")
    sampler = StratifiedSampler(random_seed=42)
    X_train_ours, X_test_ours, y_train_ours, y_test_ours = sampler.stratified_split(
        X, y, test_size=0.2, shuffle=True
    )
    
    # sklearn实现
    print(f"\n【sklearn实现】")
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 对比结果
    print(f"\n【结果对比】")
    print(f"我们的实现: 训练集{len(X_train_ours)}, 测试集{len(X_test_ours)}")
    print(f"sklearn实现: 训练集{len(X_train_sk)}, 测试集{len(X_test_sk)}")
    
    # 分布对比
    def get_distribution(y):
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {label: count / total for label, count in zip(unique, counts)}
    
    our_test_dist = get_distribution(y_test_ours)
    sk_test_dist = get_distribution(y_test_sk)
    
    print(f"\n测试集分布对比:")
    for label in sorted(our_test_dist.keys()):
        our_prop = our_test_dist[label]
        sk_prop = sk_test_dist[label]
        print(f"  类别{label}: 我们{our_prop:.1%}, sklearn{sk_prop:.1%}, 差异{abs(our_prop-sk_prop):.1%}")

def load_user_data_example():
    """
    使用实际用户数据测试算法
    """
    print("\n" + "="*60)
    print("实际用户数据测试")
    print("="*60)
    
    try:
        # 加载用户数据
        df = pd.read_csv('./output/user_profiles.csv')
        print(f"加载用户数据: {len(df)} 条记录")
        
        # 准备特征和标签
        X = df.drop(['用户ID', '消费水平'], axis=1)
        y = df['消费水平'].values
        
        # 处理分类特征 - 简单的标签编码
        from sklearn.preprocessing import LabelEncoder
        le_dict = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            le_dict[col] = le
        
        X = X.values
        
        print(f"特征矩阵: {X.shape}")
        print(f"目标变量分布: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # 执行分层抽样
        sampler = StratifiedSampler(random_seed=42)
        X_train, X_test, y_train, y_test = sampler.stratified_split(
            X, y, test_size=0.2, shuffle=True
        )
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError:
        print("未找到用户数据文件，跳过实际数据测试")
        return None, None, None, None

def algorithm_complexity_analysis():
    """
    算法复杂度分析和性能测试
    """
    print("\n" + "="*60)
    print("算法复杂度分析")
    print("="*60)
    
    print("""
算法复杂度分析:

时间复杂度:
1. 构建分层索引: O(n) - 遍历所有样本
2. 计算抽样数量: O(k) - k为类别数
3. 分层抽样: O(n log n) - 主要是随机打乱
4. 构造结果: O(n) - 索引提取
总体: O(n log n)

空间复杂度:
1. 分层索引存储: O(n) - 存储所有索引
2. 抽样计划: O(k) - 存储计划信息
3. 结果数据集: O(n) - 复制数据
总体: O(n)

优化策略:
1. 如果不需要shuffle，可降至O(n)
2. 原地操作可以减少空间复杂度
3. 并行处理可以加速大规模数据

与其他算法对比:
- 简单随机: O(n), 但无法保证分布
- 系统抽样: O(n), 但可能有周期性偏差
- 分层抽样: O(n log n), 但分布最稳定
    """)

def main():
    """
    主函数：演示分层抽样算法的完整实现
    """
    print("分层抽样算法实现 - 机器学习面试高频题")
    print("="*50)
    
    # 1. 算法复杂度分析
    algorithm_complexity_analysis()
    
    # 2. 与sklearn对比验证
    compare_with_sklearn()
    
    # 3. 实际数据测试
    X_train, X_test, y_train, y_test = load_user_data_example()
    
    # 4. 面试问题总结
    print("\n" + "="*60)
    print("面试常见问题总结")
    print("="*60)
    print("""
Q1: 分层抽样的核心思想是什么？
A1: 按目标变量分组，每组内按比例抽样，保持整体分布一致性

Q2: 时间复杂度为什么是O(n log n)？
A2: 主要瓶颈在随机打乱操作，如果不需要shuffle可降至O(n)

Q3: 与简单随机抽样的区别？
A3: 简单随机无法保证分布一致性，分层抽样专门解决这个问题

Q4: 如何处理类别不平衡问题？
A4: 确保每层至少保留一个样本，必要时调整抽样比例

Q5: 算法的局限性？
A5: 需要事先知道分层变量，对连续变量需要先离散化

Q6: 实际工程中的优化？
A6: 
   - 使用numpy向量化操作
   - 避免不必要的数据复制
   - 大数据集可考虑分块处理
   - 并行化分层内的抽样过程
    """)

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    # 运行主程序
    main()
