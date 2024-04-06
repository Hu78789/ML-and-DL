# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 创建示例数据集
X, y = make_circles(n_samples=100, noise=0.1, random_state=42, factor=0.5)

# 将数据转换为DataFrame
df = pd.DataFrame(data=X, columns=['Feature 1', 'Feature 2'])
df['Target'] = y

# 可视化数据
sns.scatterplot(x='Feature 1', y='Feature 2', hue='Target', data=df, palette='Set1')
plt.title('Scatter plot of the dataset')
plt.show()

# 将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建支持向量机分类器模型
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# 在训练集上进行预测
y_pred_train = svm_model.predict(X_train)

# 在测试集上进行预测
y_pred_test = svm_model.predict(X_test)

# 计算训练集和测试集的准确率
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")


# 绘制决策边界和支持向量
def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='Set1')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格来评估模型
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # 绘制决策边界和间隔
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

    plt.title('Decision Boundary of SVM with RBF Kernel')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


# 绘制决策边界和支持向量
plot_decision_boundary(svm_model, X, y)


