import numpy as np
import matplotlib.pyplot as plt


class BP:

    def __init__(self, layers, activation_function='sigmoid'):
        """
        初始化BP神经网络。
        :param layers: 列表，定义每层神经元数量
        :param activation_function: 激活函数，默认为sigmoid，支持 'sigmoid', 'relu', 'tanh'
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.activation_function = activation_function

        # 初始化权重和偏置
        self.weights = []
        self.biases = []

        for i in range(1, self.num_layers):
            weight = np.random.randn(layers[i],
                                     layers[i - 1])
            bias = np.random.randn(layers[i], 1)
            self.weights.append(weight)
            self.biases.append(bias)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_derivative(self, z):
        return 1 - np.tanh(z)**2

    def _get_activation(self, function_name):
        """根据函数名称返回对应的激活函数和导数"""
        if function_name == 'sigmoid':
            return self._sigmoid, self._sigmoid_derivative
        elif function_name == 'relu':
            return self._relu, self._relu_derivative
        elif function_name == 'tanh':
            return self._tanh, self._tanh_derivative
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        """
        前向传播
        :param X: 输入数据
        :return: 输出结果
        """
        self.a = [X.T]  # 存储每层的激活值
        self.z = []  # 存储每层的加权输入值
        activation, _ = self._get_activation(self.activation_function)

        # for i in range(self.num_layers - 1):
        #     z = np.dot(self.weights[i], self.a[i]) + self.biases[i]
        #     self.z.append(z)
        #     a = activation(z)
        #     self.a.append(a)

        for i in range(self.num_layers - 2):
            z = np.dot(self.weights[i], self.a[i]) + self.biases[i]
            self.z.append(z)
            a = activation(z)
            self.a.append(a)

        # 最后一层不添加激活函数
        i=self.num_layers-2
        z = np.dot(self.weights[i], self.a[i]) + self.biases[i]
        self.z.append(z)
        self.a.append(z)

        return self.a[-1]

    def backward(self, X, y, learning_rate=0.1):
        """
        反向传播
        :param X: 输入数据
        :param y: 目标标签
        :param learning_rate: 学习率
        """
        m = X.shape[0]

        # 计算误差
        delta = self.a[-1] - y.T  # 输出层误差
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        activation, activation_derivative = self._get_activation(
            self.activation_function)

        # 反向传播
        for l in range(self.num_layers - 2, -1, -1):
            dA = delta * activation_derivative(self.z[l])
            dW[l] = np.dot(dA, self.a[l].T) / m
            db[l] = np.sum(dA, axis=1, keepdims=True) / m
            delta = np.dot(self.weights[l].T, dA)

        # 更新权重和偏置
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        """
        训练神经网络
        :param X: 输入数据
        :param y: 标签
        :param epochs: 训练轮数
        :param learning_rate: 学习率
        """
        for epoch in range(epochs):
            self.forward(X)  # 前向传播
            self.backward(X, y, learning_rate)  # 反向传播
            if epoch % 100 == 0:
                loss = np.mean((self.a[-1] - y.T)**2)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        """
        预测输入数据的输出
        :param X: 输入数据
        :return: 预测的结果
        """
        output = self.forward(X)
        return output


if __name__ == "__main__":
    # 简单的数据集
    # X = np.linspace(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
    # Y = np.sin(X)
    X=((np.random.rand(100)-0.5)*2*np.pi).reshape(-1, 1)
    Y=np.sin(X)

    # 创建BP网络实例，定义层结构，激活函数为relu
    bp = BP([1, 25, 1], activation_function='sigmoid')

    # 训练网络
    bp.train(X, Y, epochs=2000, learning_rate=0.3)

    # 测试预测准确度
    predictions = bp.predict(X)
    accuracy = np.mean(predictions == Y)
    # print("Predictions:", predictions.T)
    # print(f"预测准确度: {accuracy * 100}%")
    # 绘制拟合结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X,Y,label="True Function (sin(x))", color="blue")
    plt.scatter(X,predictions,label="Predicted Function",color="red",linestyle="--")
    # plt.plot(X, Y, label="True Function (sin(x))", color="blue")
    # plt.plot(X,
    #          predictions.T,
    #          label="Predicted Function",
    #          color="red",
    #          linestyle="--")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    plt.title("Fitting sin(x) with Backpropagation Neural Network")
    plt.savefig("sin_function_fitting.png")
