#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
练习2：简单留出法体验

目标：体验简单随机划分可能带来的数据分布偏差问题
重点：观察训练集/测试集与原始数据分布的差异

作者：YjTech
版本：1.0
日期：2025年6月29日
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持 - macOS 优化版本
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',           # macOS 默认中文字体，推荐首选
    'Hiragino Sans GB',      # 冬青黑体简体中文
    'STHeiti',               # 华文黑体
    'Microsoft YaHei',       # 微软雅黑（如果安装了Office）
    'Arial Unicode MS',      # 支持Unicode的Arial
    'Heiti TC',              # 黑体-繁
    'SimHei',                # 黑体（Windows兼容）
    'DejaVu Sans'            # 开源字体备选
]
plt.rcParams['axes.unicode_minus'] = False

def load_user_data():
    """
    加载用户画像数据
    Returns:
        pd.DataFrame: 用户数据
    """
    try:
        # 尝试加载已生成的CSV文件
        df = pd.read_csv('./output/user_profiles.csv')
        print(f"成功加载用户数据，共 {len(df)} 条记录")
        return df
    except FileNotFoundError:
        print("未找到用户数据文件，请先运行 Exercise-1.py 生成数据")
        return None

def calculate_distribution(data, column_name, dataset_name="数据集"):
    """
    计算指定列的分布比例
    Args:
        data: DataFrame
        column_name: 列名
        dataset_name: 数据集名称
    Returns:
        dict: 分布比例字典
    """
    counts = data[column_name].value_counts()
    proportions = data[column_name].value_counts(normalize=True)
    
    print(f"\n{dataset_name} - {column_name}分布:")
    distribution = {}
    for value in ['高', '中', '低']:  # 确保按固定顺序显示
        if value in counts:
            count = counts[value]
            prop = proportions[value]
            distribution[value] = prop
            print(f"  {value}: {count}人 ({prop:.1%})")
        else:
            distribution[value] = 0.0
            print(f"  {value}: 0人 (0.0%)")
    
    return distribution

def calculate_deviation(original_dist, new_dist):
    """
    计算分布偏差
    Args:
        original_dist: 原始分布
        new_dist: 新分布
    Returns:
        dict: 偏差信息
    """
    deviations = {}
    print(f"\n偏差分析:")
    total_deviation = 0
    
    for level in ['高', '中', '低']:
        original_prop = original_dist.get(level, 0)
        new_prop = new_dist.get(level, 0)
        deviation = new_prop - original_prop
        deviations[level] = deviation
        total_deviation += abs(deviation)
        
        print(f"  {level}消费: {original_prop:.1%} → {new_prop:.1%} (偏差: {deviation:+.1%})")
    
    print(f"  总偏差: {total_deviation:.1%}")
    return deviations, total_deviation

def simple_holdout_experiment(df):
    """
    简单留出法实验
    Args:
        df: 原始数据
    """
    print("="*60)
    print("练习2：简单留出法体验")
    print("="*60)
    
    # 1. 统计原始数据的"黄金标准"分布
    print("\n【步骤1】建立黄金标准 - 原始数据消费水平分布")
    original_dist = calculate_distribution(df, '消费水平', "原始数据集")
    
    # 2. 进行简单随机划分（80/20）
    print("\n【步骤2】简单随机划分 (80/20)")
    # DataFrame.drop(labels, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
    # 原始数据包含所有列：['用户ID', '性别', '所在城市', '消费水平', '年龄', '最近活跃天数']
    # - label:'消费水平' - 要删除的列名
    # - axis:1 - 删除列（0表示删除行）
    # 执行后：
    # X 包含特征列：['用户ID', '性别', '所在城市', '年龄', '最近活跃天数']
    X = df.drop('消费水平', axis=1)  # 特征
    y = df['消费水平']               # 目标变量
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # 测试集占比 20%
        random_state=42     # 设置随机种子确保结果可重现
    )
    
    print(f"划分结果:")
    print(f"  训练集: {len(X_train)} 条记录 ({len(X_train)/len(df):.1%})")
    print(f"  测试集: {len(X_test)} 条记录 ({len(X_test)/len(df):.1%})")
    
    # 3. 统计训练集分布
    print("\n【步骤3】训练集消费水平分布")
    train_dist = calculate_distribution(pd.DataFrame({'消费水平': y_train}), '消费水平', "训练集")
    
    # 4. 统计测试集分布
    print("\n【步骤4】测试集消费水平分布")
    test_dist = calculate_distribution(pd.DataFrame({'消费水平': y_test}), '消费水平', "测试集")
    
    # 5. 对比分析
    print("\n【步骤5】偏差对比分析")
    print("\n训练集 vs 原始数据:")
    train_deviations, train_total_dev = calculate_deviation(original_dist, train_dist)
    
    print("\n测试集 vs 原始数据:")
    test_deviations, test_total_dev = calculate_deviation(original_dist, test_dist)
    
    # 6. 风险分析
    print("\n【步骤6】风险分析与思考")
    print("\n潜在问题:")
    
    if train_total_dev > 0.05:  # 如果总偏差超过5%
        print("  ⚠️  训练集分布偏差较大，可能导致：")
        print("     - 模型学习到偏斜的数据分布")
        print("     - 对某些消费水平的预测能力不足")
    
    if test_total_dev > 0.05:
        print("  ⚠️  测试集分布偏差较大，可能导致：")
        print("     - 模型评估结果不可靠")
        print("     - 过估计或低估模型性能")
    
    if abs(train_total_dev - test_total_dev) > 0.03:
        print("  ⚠️  训练集与测试集偏差不一致，可能导致：")
        print("     - 训练-测试分布不匹配(没有满足独立同分布的假设)")
        print("     - 模型泛化能力评估失真")
    
    if train_total_dev <= 0.02 and test_total_dev <= 0.02:
        print("  ✅ 分布偏差在可接受范围内")
    
    # 7. 可视化对比
    visualize_distribution_comparison(original_dist, train_dist, test_dist, "简单留出法")
    
    return original_dist, train_dist, test_dist

def visualize_distribution_comparison(original_dist, train_dist, test_dist, method_name):
    """
    可视化分布对比
    """
    # 准备数据
    levels = ['高', '中', '低']
    original_props = [original_dist[level] for level in levels]
    train_props = [train_dist[level] for level in levels]
    test_props = [test_dist[level] for level in levels]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 条形图对比
    x = np.arange(len(levels))
    width = 0.25
    
    ax1.bar(x - width, original_props, width, label='原始数据', color='skyblue', alpha=0.8)
    ax1.bar(x, train_props, width, label='训练集', color='lightgreen', alpha=0.8)
    ax1.bar(x + width, test_props, width, label='测试集', color='salmon', alpha=0.8)
    
    ax1.set_xlabel('消费水平')
    ax1.set_ylabel('比例')
    ax1.set_title(f'{method_name} - 消费水平分布对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (orig, train, test) in enumerate(zip(original_props, train_props, test_props)):
        ax1.text(i - width, orig + 0.01, f'{orig:.1%}', ha='center', va='bottom', fontsize=9)
        ax1.text(i, train + 0.01, f'{train:.1%}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width, test + 0.01, f'{test:.1%}', ha='center', va='bottom', fontsize=9)
    
    # 2. 偏差热力图
    deviations_data = []
    datasets = ['训练集', '测试集']
    
    for dataset, dist in [('训练集', train_dist), ('测试集', test_dist)]:
        row = []
        for level in levels:
            deviation = dist[level] - original_dist[level]
            row.append(deviation)
        deviations_data.append(row)
    
    deviations_df = pd.DataFrame(deviations_data, columns=levels, index=datasets)
    
    sns.heatmap(deviations_df, annot=True, fmt='.1%', cmap='RdBu_r', center=0,
                ax=ax2, cbar_kws={'label': '偏差 (百分点)'})
    ax2.set_title(f'{method_name} - 分布偏差热力图')
    ax2.set_xlabel('消费水平')
    ax2.set_ylabel('数据集')
    
    plt.tight_layout()
    plt.savefig(f'./output/{method_name.replace(" ", "_")}_distribution_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("简单留出法数据划分实验")
    print("="*40)
    
    # 加载数据
    df = load_user_data()
    
    if df is not None:
        # 执行简单留出法实验
        original_dist, train_dist, test_dist = simple_holdout_experiment(df)
        
        # 总结
        print("\n" + "="*60)
        print("实验总结")
        print("="*60)
        print("""
简单留出法的特点:
✅ 优点：实现简单，计算快速
❌ 缺点：可能导致数据分布偏差

关键发现：
1. 简单随机划分不能保证分布一致性
2. 小数据集更容易出现分布偏差
3. 类别不平衡时问题更明显

建议：
- 对于关键类别特征，考虑使用分层抽样
- 在模型评估时关注分布一致性
- 多次划分取平均可以降低偏差影响
        """)

if __name__ == "__main__":
    # 设置随机种子确保结果可重现
    np.random.seed(42)
    
    # 运行实验
    main()
