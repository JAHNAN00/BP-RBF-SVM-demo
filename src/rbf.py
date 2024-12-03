import numpy as np

class RBF:
    def __init__(self, centers=None, kernel='gaussian', gamma=1.0):
        """
        初始化 RBF 模型
        :param centers: 中心点，默认 None（通过 K-Means 确定）
        :param kernel: 核函数类型，默认 'gaussian'
        :param gamma: 高斯核参数，控制函数宽度
        """
        self.centers = centers
        self.kernel = kernel
        self.gamma = gamma
        self.weights = None

    def _rbf_kernel(self, x, c):
        """径向基函数"""
        if self.kernel == 'gaussian':
            return np.exp(-self.gamma * np.linalg.norm(x - c) ** 2)
        elif self.kernel == 'thin_plate_spline':
            r = np.linalg.norm(x - c)
            return r ** 2 * np.log(r + 1e-10)  # 避免 log(0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def _compute_phi(self, X):
        """计算径向基核矩阵"""
        N, M = X.shape[0], self.centers.shape[0]
        Phi = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                Phi[i, j] = self._rbf_kernel(X[i], self.centers[j])
        return Phi

    def _kmeans(self, X, k, max_iter=100, tol=1e-4):
        """
        简单实现 K-Means 聚类算法
        :param X: 输入数据 (N, D)
        :param k: 聚类数量
        :param max_iter: 最大迭代次数
        :param tol: 停止迭代的中心变化阈值
        :return: 聚类中心
        """
        # 随机初始化中心点
        indices = np.random.choice(X.shape[0], k, replace=False)
        centers = X[indices]

        for _ in range(max_iter):
            # 计算每个样本到各个中心的距离
            distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # 更新中心点
            new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])
            
            # 检查收敛
            if np.linalg.norm(new_centers - centers) < tol:
                break
            centers = new_centers

        return centers

    def _least_squares(self, Phi, y):
        """
        使用矩阵运算实现最小二乘法
        :param Phi: 核矩阵
        :param y: 目标值
        :return: 权重
        """
        return np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y

    def fit(self, X, y, num_centers=None):
        """
        拟合模型
        :param X: 训练数据 (N, D)
        :param y: 目标值 (N,)
        :param num_centers: 中心点数量，默认 None（取所有数据点）
        """
        if self.centers is None:
            if num_centers is None:
                self.centers = X
            else:
                self.centers = self._kmeans(X, num_centers)
        
        Phi = self._compute_phi(X)
        self.weights = self._least_squares(Phi, y)

    def predict(self, X):
        """
        使用拟合好的模型进行预测
        :param X: 输入数据 (N, D)
        :return: 预测值 (N,)
        """
        Phi = self._compute_phi(X)
        return Phi @ self.weights


# 测试模型
if __name__ == "__main__":
    # 生成一些数据
    X_train = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, X_train.shape[0])

    # 创建并训练模型
    rbf = RBF(kernel='gaussian', gamma=0.1)
    rbf.fit(X_train, y_train, num_centers=10)

    # 预测
    X_test = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_pred = rbf.predict(X_test)

    # 可视化
    import matplotlib.pyplot as plt
    plt.scatter(X_train, y_train, label="Training data", color="blue")
    plt.plot(X_test, y_pred, label="RBF Prediction", color="red")
    plt.legend()
    #plt.show()
    plt.savefig("sin_function_fitting.png")
