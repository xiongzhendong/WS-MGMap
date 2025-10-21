import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ==================== 1. 加载并预处理数据 ====================
print("--- 1. 数据加载与预处理 ---")
# 使用 TensorFlow/Keras 加载 MNIST 数据集，这种方式更稳定
(x_train_keras, y_train_keras), (x_test_keras, y_test_keras) = tf.keras.datasets.mnist.load_data()

# 合并 Keras 的训练集和测试集，以创建一个统一的数据池用于采样
x_all = np.vstack((x_train_keras, x_test_keras))
y_all = np.concatenate((y_train_keras, y_test_keras))

# 将 (28, 28) 的二维图像展平为 784 维向量
x_all_flattened = x_all.reshape(-1, 28 * 28).astype(np.float64)
y_all = y_all.astype(np.int64)

# 从完整数据集中为每个数字类别（0-9）筛选 100 个样本
selected_x_list = []
selected_y_list = []
for digit in range(10):
    idx = np.where(y_all == digit)[0][:100]
    selected_x_list.append(x_all_flattened[idx])
    selected_y_list.append(y_all[idx])

# 将样本列表合并成最终的 NumPy 数组
X = np.vstack(selected_x_list)
y = np.concatenate(selected_y_list)

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y # stratify确保训练/测试集中类别比例一致
)

# 特征标准化：在训练集上拟合，并应用到训练集和测试集
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"总样本数: {X.shape[0]}")
print(f"训练集形状: {X_train_scaled.shape}")
print(f"测试集形状: {X_test_scaled.shape}")

# ==================== 2. PCA 降维与方差分析 ====================
print("\n--- 2. PCA 降维与方差分析 ---")
# 计算保留 95% 方差所需的主成分数
# 这是确定最佳降维维度的标准方法
pca_95_variance = PCA(n_components=0.95, random_state=42)
pca_95_variance.fit(X_train_scaled)
n_components_95 = pca_95_variance.n_components_
print(f"保留 95% 方差所需主成分数: {n_components_95}")

# 为了绘制平滑的曲线，我们计算更多主成分（例如100个）的方差贡献
pca_for_plot = PCA(n_components=100, random_state=42).fit(X_train_scaled)
cumulative_variance = np.cumsum(pca_for_plot.explained_variance_ratio_)

# 绘制累计方差贡献率曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), cumulative_variance, 'b-o', markersize=4)
plt.axhline(y=0.95, color='r', linestyle='--', label=f'95% 方差阈值')
plt.axvline(x=n_components_95, color='g', linestyle=':', label=f'k = {n_components_95}')
plt.xlabel('Number of main components (k)')
plt.ylabel('Cumulative variance contribution rate')
plt.title('PCA 累计方差贡献率曲线')
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig('pca_variance_curve.png')
plt.show()

# ==================== 3. 降维可视化与分类对比 ====================
print("\n--- 3. 降维可视化与分类对比 ---")
# a) 降维至 2D 可视化
pca_2d = PCA(n_components=2, random_state=42)
X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)

plt.figure(figsize=(12, 8))
cmap = plt.get_cmap('tab10', 10)
for digit in range(10):
    mask = (y_train == digit)
    plt.scatter(
        X_train_pca_2d[mask, 0], X_train_pca_2d[mask, 1],
        color=cmap(digit), label=f'数字 {digit}', alpha=0.8, s=50
    )
plt.xlabel(f'主成分 1 (方差贡献率: {pca_2d.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'主成分 2 (方差贡献率: {pca_2d.explained_variance_ratio_[1]:.2%})')
plt.title('MNIST 数据集 PCA 降维至 2D 可视化')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()

# b) 分类性能对比
# 场景一：在原始数据（784维）上训练 SVM
print("正在训练原始数据 (784维)...")
svm_original = SVC(kernel='rbf', random_state=42)
start_time = time.time()
svm_original.fit(X_train_scaled, y_train)
train_time_original = time.time() - start_time
acc_original = svm_original.score(X_test_scaled, y_test)

# 场景二：在 PCA 降维数据（保留95%方差）上训练 SVM
print(f"正在训练降维数据 ({n_components_95}维)...")
# 使用之前已经拟合好的 pca_95_variance 来转换数据
X_train_pca = pca_95_variance.transform(X_train_scaled)
X_test_pca = pca_95_variance.transform(X_test_scaled)

svm_pca = SVC(kernel='rbf', random_state=42)
start_time = time.time()
svm_pca.fit(X_train_pca, y_train)
train_time_pca = time.time() - start_time
acc_pca = svm_pca.score(X_test_pca, y_test)

# ==================== 4. 结果展示 ====================
print("\n--- 4. 实验结果对比 ---")
print(f"原始数据 (784维):")
print(f"  - 测试集准确率: {acc_original:.4f}")
print(f"  - 模型训练时间: {train_time_original:.4f} 秒")
print("-" * 30)
print(f"PCA 降维数据 ({n_components_95}维):")
print(f"  - 测试集准确率: {acc_pca:.4f}")
print(f"  - 模型训练时间: {train_time_pca:.4f} 秒")
print("-" * 30)
print("性能分析:")
reduction_ratio = (1 - n_components_95 / X_train_scaled.shape[1]) * 100
print(f"  - 维度压缩率: {reduction_ratio:.2f}%")
if train_time_pca > 0:
    speedup_ratio = train_time_original / train_time_pca
    print(f"  - 训练加速比: {speedup_ratio:.2f} 倍")