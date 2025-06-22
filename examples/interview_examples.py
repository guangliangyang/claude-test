"""
Canva AI Engineer Interview - Ready-to-use Examples
这些例子涵盖了常见的AI/ML面试题目，可以直接运行演示
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split as sklearn_split

# Import our custom implementations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_algorithms import LinearRegression, KMeans, NaiveBayes, train_test_split, accuracy_score, mean_squared_error
from src.data_processing import DataProcessor, create_sample_dataset, create_classification_dataset


def example_1_linear_regression():
    """
    示例1: 线性回归 - 房价预测
    常见面试题: 实现并测试线性回归算法
    """
    print("=" * 50)
    print("示例1: 线性回归 - 房价预测")
    print("=" * 50)
    
    # 创建模拟房价数据
    np.random.seed(42)
    # 特征: 面积(平方米), 房间数, 楼层, 距离市中心距离(公里)
    X = np.random.rand(200, 4)
    X[:, 0] = X[:, 0] * 200 + 50   # 面积: 50-250平方米
    X[:, 1] = X[:, 1] * 4 + 1      # 房间数: 1-5个
    X[:, 2] = X[:, 2] * 20 + 1     # 楼层: 1-21层
    X[:, 3] = X[:, 3] * 30 + 1     # 距离: 1-31公里
    
    # 房价 = 面积*3000 + 房间数*50000 + 楼层*2000 - 距离*5000 + 噪声
    y = (X[:, 0] * 3000 + X[:, 1] * 50000 + X[:, 2] * 2000 - X[:, 3] * 5000 + 
         np.random.normal(0, 20000, 200))
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = LinearRegression(learning_rate=0.0000001, max_iterations=1000)
    model.fit(X_train, y_train)
    
    # 预测
    predictions = model.predict(X_test)
    
    # 评估
    mse = mean_squared_error(y_test, predictions)
    print(f"均方误差: {mse:,.0f}")
    print(f"预测示例:")
    for i in range(5):
        print(f"  实际价格: ¥{y_test[i]:,.0f}, 预测价格: ¥{predictions[i]:,.0f}")
    
    return model, X_test, y_test, predictions


def example_2_kmeans_clustering():
    """
    示例2: K-Means聚类 - 客户分群
    常见面试题: 实现聚类算法进行客户分析
    """
    print("\n" + "=" * 50)
    print("示例2: K-Means聚类 - 客户分群")
    print("=" * 50)
    
    # 创建客户数据
    np.random.seed(42)
    # 特征: 年收入(万), 年消费(万)
    
    # 高收入高消费群体
    high_income = np.random.normal([80, 60], [15, 10], (50, 2))
    # 中等收入中等消费群体
    mid_income = np.random.normal([50, 35], [10, 8], (50, 2))
    # 低收入低消费群体
    low_income = np.random.normal([25, 15], [8, 5], (50, 2))
    
    X = np.vstack([high_income, mid_income, low_income])
    
    # 聚类
    kmeans = KMeans(k=3, random_state=42)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    
    print(f"聚类中心:")
    for i, center in enumerate(kmeans.centroids):
        print(f"  群体{i+1}: 年收入{center[0]:.1f}万, 年消费{center[1]:.1f}万")
    
    print(f"\n各群体客户数量:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  群体{cluster+1}: {count}人")
    
    return kmeans, X, labels


def example_3_naive_bayes_classification():
    """
    示例3: 朴素贝叶斯分类 - 邮件垃圾检测
    常见面试题: 实现分类算法
    """
    print("\n" + "=" * 50)
    print("示例3: 朴素贝叶斯分类 - 邮件分类")
    print("=" * 50)
    
    # 创建邮件特征数据
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, 
                              n_informative=3, n_redundant=1, random_state=42)
    
    # 特征含义: [关键词频率, 链接数量, 大写字母比例, 感叹号数量]
    feature_names = ['关键词频率', '链接数量', '大写字母比例', '感叹号数量']
    class_names = ['正常邮件', '垃圾邮件']
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    
    # 预测
    predictions = nb.predict(X_test)
    probabilities = nb.predict_proba(X_test)
    
    # 评估
    acc = accuracy_score(y_test, predictions)
    print(f"分类准确率: {acc:.3f}")
    
    print(f"\n预测示例:")
    for i in range(5):
        pred_class = class_names[predictions[i]]
        confidence = probabilities[i][predictions[i]]
        print(f"  邮件{i+1}: {pred_class} (置信度: {confidence:.3f})")
    
    return nb, X_test, y_test, predictions


def example_4_data_preprocessing():
    """
    示例4: 数据预处理流水线
    常见面试题: 处理真实世界的脏数据
    """
    print("\n" + "=" * 50)
    print("示例4: 数据预处理流水线")
    print("=" * 50)
    
    # 创建带有缺失值和异常值的数据
    np.random.seed(42)
    data = {
        'age': [25, 30, np.nan, 40, 35, 28, 45, np.nan, 32, 29, 100],  # 100是异常值
        'income': [50000, 60000, 55000, 80000, np.nan, 52000, 90000, 65000, 58000, 54000, 200000],
        'education': ['本科', '硕士', '本科', '博士', '硕士', '本科', '博士', '硕士', '本科', '博士', '本科'],
        'city': ['北京', '上海', '北京', '深圳', '上海', '北京', '深圳', '上海', '北京', '深圳', '广州']
    }
    df = pd.DataFrame(data)
    
    print("原始数据:")
    print(df.head())
    print(f"\n缺失值统计:")
    print(df.isnull().sum())
    
    # 数据预处理
    processor = DataProcessor()
    
    # 1. 处理缺失值
    df_filled = processor.handle_missing_values(df, strategy='mean', columns=['age', 'income'])
    print(f"\n处理缺失值后:")
    print(f"Age缺失值: {df_filled['age'].isnull().sum()}")
    print(f"Income缺失值: {df_filled['income'].isnull().sum()}")
    
    # 2. 移除异常值
    df_clean = processor.remove_outliers(df_filled, columns=['age', 'income'], method='iqr')
    print(f"\n移除异常值后数据量: {len(df)} -> {len(df_clean)}")
    
    # 3. 编码分类变量
    df_encoded = processor.encode_categorical(df_clean, columns=['education', 'city'], method='onehot')
    print(f"\n编码后特征数: {len(df_clean.columns)} -> {len(df_encoded.columns)}")
    print(f"新特征: {[col for col in df_encoded.columns if col.startswith(('education_', 'city_'))]}")
    
    # 4. 标准化数值特征
    df_scaled = processor.scale_features(df_encoded, columns=['age', 'income'], method='standard')
    print(f"\n标准化后的age均值: {df_scaled['age'].mean():.6f}")
    print(f"标准化后的age标准差: {df_scaled['age'].std():.6f}")
    
    return processor, df, df_scaled


def example_5_model_comparison():
    """
    示例5: 模型对比实验
    常见面试题: 比较不同算法的性能
    """
    print("\n" + "=" * 50)
    print("示例5: 模型对比实验")
    print("=" * 50)
    
    # 创建分类数据集
    X, y = create_classification_dataset(n_samples=500, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 我们的朴素贝叶斯
    nb_custom = NaiveBayes()
    nb_custom.fit(X_train, y_train)
    nb_pred = nb_custom.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    
    # sklearn的朴素贝叶斯作为对比
    from sklearn.naive_bayes import GaussianNB
    nb_sklearn = GaussianNB()
    nb_sklearn.fit(X_train, y_train)
    nb_sklearn_pred = nb_sklearn.predict(X_test)
    nb_sklearn_acc = accuracy_score(y_test, nb_sklearn_pred)
    
    print(f"模型性能对比:")
    print(f"  自实现朴素贝叶斯: {nb_acc:.3f}")
    print(f"  Sklearn朴素贝叶斯:  {nb_sklearn_acc:.3f}")
    print(f"  性能差异: {abs(nb_acc - nb_sklearn_acc):.3f}")
    
    return nb_custom, nb_sklearn, X_test, y_test


def run_all_examples():
    """运行所有示例"""
    print("Canva AI Engineer Interview - 演示示例")
    print("时间:", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 运行所有示例
    model1, X_test1, y_test1, pred1 = example_1_linear_regression()
    model2, X2, labels2 = example_2_kmeans_clustering()  
    model3, X_test3, y_test3, pred3 = example_3_naive_bayes_classification()
    processor, df_orig, df_final = example_4_data_preprocessing()
    nb_custom, nb_sklearn, X_test5, y_test5 = example_5_model_comparison()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成！")
    print("面试环境准备就绪 ✅")
    print("=" * 50)


if __name__ == "__main__":
    run_all_examples()