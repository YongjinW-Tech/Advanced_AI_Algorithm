#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
练习1：用户画像的模拟与生成

目标：通过编程自动生成一批"用户画像"数据，掌握数据合成与基本属性建模方法
应用场景：推荐系统、用户分析、机器学习建模等

作者：YjTech
版本：1.0
日期：2025年6月29日
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持 - macOS 优化版本
# 以下字体按优先级排列，matplotlib 会依次尝试
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
# 设置负号显示正常
plt.rcParams['axes.unicode_minus'] = False

class UserProfileGenerator:
    """用户画像生成器类"""

    def __init__(self):
        """初始化用户画像生成器，定义各属性的取值范围和分布"""
        
        # 定义类别型属性的取值集合和权重
        self.categorical_attributes = {
            '性别': {
                'values': ['男', '女', '未透露'],
                'weights': [0.48, 0.47, 0.05]  # 男性48%，女性47%，未透露5%
            },
            '所在城市': {
                'values': ['北京', '上海', '广州', '深圳', '湛江', '其他'],
                'weights': [0.15, 0.12, 0.08, 0.10, 0.05, 0.50]  # 一线城市占45%，其他55%
            },
            '消费水平': {
                'values': ['高', '中', '低'],
                'weights': [0.20, 0.60, 0.20]  # 中等消费水平占主体60%
            }
        }
        
        # 定义数值型属性的取值范围
        self.numerical_attributes = {
            '年龄': {
                'min': 16,
                'max': 65,
                'distribution': 'normal',  # 正态分布
                'mean': 32,
                'std': 12
            },
            '最近活跃天数': {
                'min': 0,
                'max': 365,
                'distribution': 'exponential',  # 指数分布，模拟用户活跃度递减
                'lambda': 0.02
            }
        }
    
    def generate_categorical_value(self, attribute_name: str) -> str:
        """
        生成类别型属性
        Args:
            attribute_name: 属性名称
        Returns:
            随机生成的属性值
        """
        attr_config = self.categorical_attributes[attribute_name]
        # np.random.choice(a, size=None, replace=True, p=None)
        # - size 参数：控制返回结果的数量和形状；size = None（默认），返回单个值；size = 整数，返回一维数组；size = 元组，返回多维数组
        # - replace 参数：是否允许重复抽样，当 replace=False 时，size 不能超过候选值的数量
        return np.random.choice(
            attr_config['values'], 
            p=attr_config['weights']
        )
    
    def generate_numerical_value(self, attribute_name: str) -> int:
        """
        生成数值型属性值
        Args:
            attribute_name: 属性名称
        Returns:
            随机生成的数值
        """
        attr_config = self.numerical_attributes[attribute_name]
        
        if attr_config['distribution'] == 'normal':
            # 正态分布生成年龄
            value = np.random.normal(attr_config['mean'], attr_config['std'])
            # 确保值在合理范围内
            value = max(attr_config['min'], min(attr_config['max'], int(value)))
            
        elif attr_config['distribution'] == 'exponential':
            # 指数分布生成活跃天数（大多数用户近期活跃）
            value = np.random.exponential(1/attr_config['lambda'])
            value = max(attr_config['min'], min(attr_config['max'], int(value)))
            
        return value
    
    def generate_single_user(self, user_id: int) -> Dict[str, Any]:
        """
        生成单个用户的画像数据
        Args:
            user_id: 用户ID
        Returns:
            包含用户所有属性的字典
        """
        user_profile = {'用户ID': f'USER_{user_id:05d}'}
        
        # 生成类别型属性
        for attr_name in self.categorical_attributes.keys():
            user_profile[attr_name] = self.generate_categorical_value(attr_name)
        
        # 生成数值型属性
        for attr_name in self.numerical_attributes.keys():
            user_profile[attr_name] = self.generate_numerical_value(attr_name)
        
        # 添加一些业务逻辑关联性
        # 例如：高消费水平的用户年龄可能偏大一些
        if user_profile['消费水平'] == '高' and random.random() < 0.3:
            user_profile['年龄'] = min(60, user_profile['年龄'] + random.randint(5, 15))
        
        # 一线城市用户消费水平可能偏高
        if user_profile['所在城市'] in ['北京', '上海', '深圳'] and random.random() < 0.2:
            if user_profile['消费水平'] == '低':
                user_profile['消费水平'] = '中'
        
        return user_profile
    
    def generate_users_batch(self, num_users: int = 500) -> pd.DataFrame:
        """
        批量生成用户画像数据
        Args:
            num_users: 要生成的用户数量
        Returns:
            包含所有用户数据的 DataFrame
        """
        print(f"开始生成 {num_users} 个用户画像...")
        
        users_data = []
        for i in range(1, num_users + 1):
            user_profile = self.generate_single_user(i)
            users_data.append(user_profile)
        
        df = pd.DataFrame(users_data)
        print(f"用户画像生成完成！共生成 {len(df)} 个用户")
        
        return df
    
    def analyze_generated_data(self, df: pd.DataFrame) -> None:
        """
        分析生成的用户画像数据
        Args:
            df: 用户画像数据DataFrame
        """
        print("\n" + "="*50)
        print("用户画像数据分析报告")
        print("="*50)
        
        # 基本统计信息
        print(f"\n1. 数据概览:")
        print(f"   总用户数: {len(df)}")
        print(f"   属性数量: {len(df.columns)}")
        print(f"   数据维度: {df.shape}")
        
        # 类别型属性分布
        print(f"\n2. 类别型属性分布:")
        for col in ['性别', '所在城市', '消费水平']:
            print(f"\n   {col}分布:")
            counts = df[col].value_counts()
            for value, count in counts.items():
                percentage = (count / len(df)) * 100
                print(f"     {value}: {count}人 ({percentage:.1f}%)")
        
        # 数值型属性统计
        print(f"\n3. 数值型属性统计:")
        numerical_cols = ['年龄', '最近活跃天数']
        for col in numerical_cols:
            stats = df[col].describe()
            print(f"\n   {col}:")
            print(f"     平均值: {stats['mean']:.1f}")
            print(f"     中位数: {stats['50%']:.1f}")
            print(f"     标准差: {stats['std']:.1f}")
            print(f"     取值范围: {stats['min']:.0f} - {stats['max']:.0f}")
    
    def visualize_data(self, df: pd.DataFrame, save_plots: bool = True) -> None:
        """
        可视化用户画像数据
        Args:
            df: 用户画像数据DataFrame
            save_plots: 是否保存图表
        """
        print("\n生成数据可视化图表...")
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10)) # 创建 2 行 3 列的子图，子图大小为 15x10 英寸
        fig.suptitle('用户画像数据分析可视化', fontsize=20, fontweight='bold')
        
        # 1. 性别分布饼图
        gender_counts = df['性别'].value_counts()
        axes[0, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('性别分布')
        
        # 2. 城市分布条形图
        city_counts = df['所在城市'].value_counts()
        axes[0, 1].bar(city_counts.index, city_counts.values)
        axes[0, 1].set_title('城市分布')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 消费水平分布
        consumption_counts = df['消费水平'].value_counts()
        axes[0, 2].bar(consumption_counts.index, consumption_counts.values, 
                       color=['red', 'orange', 'green'])
        axes[0, 2].set_title('消费水平分布')
        
        # 4. 年龄分布直方图
        axes[1, 0].hist(df['年龄'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('年龄分布')
        axes[1, 0].set_xlabel('年龄')
        axes[1, 0].set_ylabel('人数')
        
        # 5. 活跃天数分布
        axes[1, 1].hist(df['最近活跃天数'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].set_title('最近活跃天数分布')
        axes[1, 1].set_xlabel('天数')
        axes[1, 1].set_ylabel('人数')
        
        # 6. 年龄vs消费水平箱线图
        consumption_order = ['低', '中', '高']
        df_plot = df[df['消费水平'].isin(consumption_order)] # 过滤出有效消费水平数据
        sns.boxplot(data=df_plot, x='消费水平', y='年龄', ax=axes[1, 2], order=consumption_order)
        axes[1, 2].set_title('不同消费水平的年龄分布')
        
        # 调整布局
        plt.tight_layout() # 确保子图之间不会重叠
        
        if save_plots:
            plt.savefig('./output/user_profile_analysis.png', dpi=300, bbox_inches='tight')
            print("图表已保存为 './output/user_profile_analysis.png'")
        
        plt.show()

def main():

    print("用户画像模拟与生成系统")
    print("="*30)
    
    # 1. 创建用户画像生成器
    generator = UserProfileGenerator()
    
    # 2. 生成用户画像数据
    num_users = 1000  # 可以调整生成的用户数量
    user_df = generator.generate_users_batch(num_users)
    
    # 3. 显示前几条数据
    print(f"\n前5个用户的画像数据:")
    print(user_df.head())
    
    # 4. 分析生成的数据
    generator.analyze_generated_data(user_df)
    
    # 5. 保存数据到CSV文件
    output_file = './output/user_profiles.csv'
    user_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n数据已保存到 '{output_file}'")
    
    # 6. 数据可视化
    try:
        generator.visualize_data(user_df)
    except Exception as e:
        print(f"可视化生成失败: {e}")
        print("可能需要安装 matplotlib 和 seaborn: pip install matplotlib seaborn")
        
    return user_df


if __name__ == "__main__":
    # 设置随机种子，确保结果可重现（可选）
    np.random.seed(42)
    random.seed(42)
    
    # 运行主程序
    generated_data = main()