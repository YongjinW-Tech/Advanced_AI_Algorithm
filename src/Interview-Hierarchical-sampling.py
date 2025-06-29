import numpy as np
from collections import defaultdict

def stratified_split_simple(X, y, test_size=0.2, random_state=42):
    """
    面试版分层抽样算法 - 核心实现
    
    Args:
        X: 特征数据 (n_samples, n_features)
        y: 目标变量 (n_samples,)
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 步骤1：按类别分组 - O(n)
    groups = defaultdict(list)      # defaultdict 是 dict 的子类，提供默认值功能：当访问不存在的key时，自动创建一个空列表
    for i, label in enumerate(y):
        groups[label].append(i)     # 将样本索引按类别分组
    
    print(f"发现 {len(groups)} 个分层:")
    for label, indices in groups.items():  # groups.items() 返回每个类别及其对应的样本索引
        print(f"  {label}: {len(indices)} 样本")
    
    # 步骤2：每组内分层抽样 - O(n)
    train_idx, test_idx = [], []
    
    for label, indices in groups.items():
        # 计算该组的测试集数量
        n_test = int(len(indices) * test_size)
        n_test = max(1, n_test) if len(indices) > 1 else 0  # 至少1个测试样本
        
        # 随机打乱并分割
        shuffled = np.random.permutation(indices)  # shuffled的类型是 ndarray
        test_idx.extend(shuffled[:n_test])   # 这里为什么使用extend而不是append？因为shuffled是一个数组，我们需要将其元素添加到test_idx中
        train_idx.extend(shuffled[n_test:])
        
        print(f"  {label}: 训练集{len(shuffled[n_test:])}, 测试集{n_test}")
    
    # 步骤3：构造结果
    train_idx = np.array(train_idx)     # 转换为NumPy数组,train_idx和test_idx是样本索引，之前是列表
    test_idx = np.array(test_idx)
    
    # 最终打乱
    train_idx = np.random.permutation(train_idx)
    test_idx = np.random.permutation(test_idx)
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def verify_distribution(y_original, y_train, y_test):
    """验证分布一致性"""
    def get_dist(y):
        unique, counts = np.unique(y, return_counts=True)
        return {k: v/len(y) for k, v in zip(unique, counts)}
    
    orig_dist = get_dist(y_original)
    train_dist = get_dist(y_train)
    test_dist = get_dist(y_test)
    
    print(f"\n分布验证:")
    print(f"{'类别':<6} {'原始':<8} {'训练':<8} {'测试':<8} {'训练偏差':<8} {'测试偏差'}")
    print("-" * 50)
    
    max_dev = 0
    for label in orig_dist:
        orig = orig_dist[label]
        train = train_dist.get(label, 0)
        test = test_dist.get(label, 0)
        train_dev = abs(train - orig)
        test_dev = abs(test - orig)
        max_dev = max(max_dev, train_dev, test_dev)
        
        print(f"{label:<6} {orig:<8.1%} {train:<8.1%} {test:<8.1%} {train_dev:<8.1%} {test_dev:<8.1%}")
    
    print(f"\n最大偏差: {max_dev:.1%}")
    return max_dev < 0.05

def main():
    print("面试版分层抽样算法演示")
    print("=" * 40)
    
    # 创建测试数据
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_classes=3, n_clusters_per_class=1, 
                              random_state=42)
    
    print(f"测试数据: {len(X)} 样本, {len(np.unique(y))} 类别")
    
    # 执行分层抽样
    X_train, X_test, y_train, y_test = stratified_split_simple(X, y, test_size=0.2)
    
    print(f"\n结果: 训练集{len(X_train)}, 测试集{len(X_test)}")
    
    # 验证效果
    is_good = verify_distribution(y, y_train, y_test)
    print(f"分层效果: {'✅ 优秀' if is_good else '⚠️ 一般'}")
    
    print(f"\n核心思想总结:")
    print(f"1. 按目标变量分组 (分层)")
    print(f"2. 每组内按比例抽样")
    print(f"3. 保持分布一致性")
    print(f"4. 时间复杂度: O(n)")


if __name__ == "__main__":
    main()
    