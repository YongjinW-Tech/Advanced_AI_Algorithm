#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
练习3：分层抽样体验

目标：体验分层抽样如何保持数据分布的一致性
重点：对比分层抽样与简单随机划分的效果差异

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
        data: DataFrame或Series
        column_name: 列名
        dataset_name: 数据集名称
    Returns:
        dict: 分布比例字典
    """
    if isinstance(data, pd.Series):
        # 如果输入是Series，直接使用
        series_data = data
    else:
        # 如果输入是DataFrame，提取对应列
        series_data = data[column_name]
    
    counts = series_data.value_counts()
    proportions = series_data.value_counts(normalize=True)
    
    print(f"\n{dataset_name} - 消费水平分布:")
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
        tuple: (偏差字典, 总偏差)
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

def stratified_sampling_experiment(df):
    """
    分层抽样实验
    Args:
        df: 原始数据
    """
    print("="*60)
    print("练习3：分层抽样体验")
    print("="*60)
    
    # 1. 统计原始数据的"黄金标准"分布
    print("\n【步骤1】黄金标准 - 原始数据消费水平分布")
    original_dist = calculate_distribution(df, '消费水平', "原始数据集")
    
    # 2. 进行分层抽样划分（80/20）
    print("\n【步骤2】分层抽样划分 (80/20)")
    X = df.drop('消费水平', axis=1)  # 特征
    y = df['消费水平']               # 目标变量
    
    # 使用stratify参数进行分层抽样
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
        X, y, 
        test_size=0.2,      # 测试集占比 20%
        random_state=42,    # 设置随机种子确保结果可重现
        stratify=y          # 按消费水平（黄金标准）分层抽样：按照 y 的分布比例进行分层抽样
    )
    
    print(f"分层划分结果:")
    print(f"  训练集: {len(X_train_strat)} 条记录 ({len(X_train_strat)/len(df):.1%})")
    print(f"  测试集: {len(X_test_strat)} 条记录 ({len(X_test_strat)/len(df):.1%})")
    
    # 3. 统计分层训练集分布
    print("\n【步骤3】分层训练集消费水平分布")
    train_strat_dist = calculate_distribution(y_train_strat, None, "分层训练集")
    
    # 4. 统计分层测试集分布
    print("\n【步骤4】分层测试集消费水平分布")
    test_strat_dist = calculate_distribution(y_test_strat, None, "分层测试集")
    
    # 5. 对比分析
    print("\n【步骤5】偏差对比分析")
    print("\n分层训练集 vs 原始数据:")
    train_strat_deviations, train_strat_total_dev = calculate_deviation(original_dist, train_strat_dist)
    
    print("\n分层测试集 vs 原始数据:")
    test_strat_deviations, test_strat_total_dev = calculate_deviation(original_dist, test_strat_dist)
    
    # 6. 与简单随机划分对比
    print("\n【步骤6】与简单随机划分对比")
    
    # 进行简单随机划分作为对比
    X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
        X, y, test_size=0.2, random_state=42  # 不使用stratify
    )
    
    train_simple_dist = calculate_distribution(y_train_simple, None, "简单随机训练集")
    test_simple_dist = calculate_distribution(y_test_simple, None, "简单随机测试集")
    
    print("\n简单随机训练集 vs 原始数据:")
    _, train_simple_total_dev = calculate_deviation(original_dist, train_simple_dist)
    
    print("\n简单随机测试集 vs 原始数据:")
    _, test_simple_total_dev = calculate_deviation(original_dist, test_simple_dist)
    
    # 7. 优势分析
    print("\n【步骤7】分层抽样优势分析")
    
    print(f"\n偏差对比总结:")
    print(f"  分层抽样 - 训练集总偏差: {train_strat_total_dev:.1%}")
    print(f"  简单随机 - 训练集总偏差: {train_simple_total_dev:.1%}")
    print(f"  训练集偏差改善: {train_simple_total_dev - train_strat_total_dev:+.1%}")
    
    print(f"\n  分层抽样 - 测试集总偏差: {test_strat_total_dev:.1%}")
    print(f"  简单随机 - 测试集总偏差: {test_simple_total_dev:.1%}")
    print(f"  测试集偏差改善: {test_simple_total_dev - test_strat_total_dev:+.1%}")
    
    # 8. 效果评估
    print("\n【步骤8】分层抽样效果评估")
    
    if train_strat_total_dev < 0.01 and test_strat_total_dev < 0.01:
        print("  ✅ 分层抽样效果优秀：分布偏差 < 1%")
    elif train_strat_total_dev < 0.02 and test_strat_total_dev < 0.02:
        print("  ✅ 分层抽样效果良好：分布偏差 < 2%")
    else:
        print("  ⚠️  分层抽样效果一般：可能需要更大的数据集")
    
    improvement_train = train_simple_total_dev - train_strat_total_dev
    improvement_test = test_simple_total_dev - test_strat_total_dev
    
    if improvement_train > 0.01 or improvement_test > 0.01:
        print("  ✅ 相比简单随机划分有显著改善")
    else:
        print("  ℹ️  改善效果有限，可能原数据分布较均匀")
    
    # 9. 可视化对比
    visualize_stratified_comparison(
        original_dist, 
        train_strat_dist, test_strat_dist,
        train_simple_dist, test_simple_dist
    )
    
    return {
        'original': original_dist,
        'stratified_train': train_strat_dist,
        'stratified_test': test_strat_dist,
        'simple_train': train_simple_dist,
        'simple_test': test_simple_dist
    }

def visualize_stratified_comparison(original_dist, train_strat_dist, test_strat_dist, 
                                  train_simple_dist, test_simple_dist):
    """
    可视化分层抽样与简单随机划分的对比
    """
    # 准备数据
    levels = ['高', '中', '低']
    original_props = [original_dist[level] for level in levels]
    train_strat_props = [train_strat_dist[level] for level in levels]
    test_strat_props = [test_strat_dist[level] for level in levels]
    train_simple_props = [train_simple_dist[level] for level in levels]
    test_simple_props = [test_simple_dist[level] for level in levels]
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('分层抽样 vs 简单随机划分 - 效果对比', fontsize=16, fontweight='bold')
    
    # 1. 训练集对比
    x = np.arange(len(levels))
    width = 0.25
    
    ax1.bar(x - width/2, original_props, width, label='原始数据', color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, train_strat_props, width, label='分层训练集', color='lightgreen', alpha=0.8)
    
    ax1.set_xlabel('消费水平')
    ax1.set_ylabel('比例')
    ax1.set_title('分层抽样 - 训练集分布')
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 测试集对比  
    ax2.bar(x - width/2, original_props, width, label='原始数据', color='skyblue', alpha=0.8)
    ax2.bar(x + width/2, test_strat_props, width, label='分层测试集', color='salmon', alpha=0.8)
    
    ax2.set_xlabel('消费水平')
    ax2.set_ylabel('比例')
    ax2.set_title('分层抽样 - 测试集分布')
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 简单随机对比
    ax3.bar(x - width/2, original_props, width, label='原始数据', color='skyblue', alpha=0.8)
    ax3.bar(x + width/2, train_simple_props, width, label='简单随机训练集', color='orange', alpha=0.8)
    
    ax3.set_xlabel('消费水平')
    ax3.set_ylabel('比例')
    ax3.set_title('简单随机划分 - 训练集分布')
    ax3.set_xticks(x)
    ax3.set_xticklabels(levels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 偏差对比热力图
    deviations_data = []
    methods = ['分层训练集', '分层测试集', '随机训练集', '随机测试集']
    distributions = [train_strat_dist, test_strat_dist, train_simple_dist, test_simple_dist]
    
    for method, dist in zip(methods, distributions):
        row = []
        for level in levels:
            deviation = dist[level] - original_dist[level]
            row.append(deviation)
        deviations_data.append(row)
    
    deviations_df = pd.DataFrame(deviations_data, columns=levels, index=methods)
    
    sns.heatmap(deviations_df, annot=True, fmt='.1%', cmap='RdBu_r', center=0,
                ax=ax4, cbar_kws={'label': '偏差 (百分点)'})
    ax4.set_title('分布偏差对比热力图')
    ax4.set_xlabel('消费水平')
    ax4.set_ylabel('划分方法')
    
    plt.tight_layout()
    plt.savefig('./output/stratified_vs_simple_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("分层抽样数据划分实验")
    print("="*40)
    
    # 加载数据
    df = load_user_data()
    
    if df is not None:
        # 执行分层抽样实验
        results = stratified_sampling_experiment(df)
        
        # 总结
        print("\n" + "="*60)
        print("实验总结")
        print("="*60)
        print("""
分层抽样的特点:
✅ 优点：保持原始数据分布
✅ 优点：降低抽样偏差
✅ 优点：提高模型评估可靠性

核心原理：
1. 按目标变量分层
2. 每层按比例抽样
3. 确保各层分布一致

适用场景：
- 类别不平衡数据
- 关键特征需要保持分布
- 对模型评估准确性要求高

建议：
- 优先选择分层抽样
- 特别是分类任务
- 监控训练/测试分布一致性
        """)
    else:
        print("数据加载失败，请检查数据文件")

if __name__ == "__main__":
    # 设置随机种子确保结果可重现
    np.random.seed(42)
    
    # 运行实验
    main()
