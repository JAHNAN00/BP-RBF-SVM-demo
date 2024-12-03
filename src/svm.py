import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


# Load dataset
def load_data(fname):
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


# Calculate classification accuracy
def eval_acc(label, pred):
    return np.sum(label == pred) / len(pred)


# Visualization
def show_data(data):
    fig, ax = plt.subplots()
    cls = data[:, 2]
    ax.scatter(data[:, 0][cls == 1], data[:, 1][cls == 1])
    ax.scatter(data[:, 0][cls == -1], data[:, 1][cls == -1])
    ax.grid(False)
    fig.tight_layout()
    plt.savefig('svm.png')


def kernel_gauss(x, sigma=1):
    if np.size(x) == 2:  # 当x为一个向量时，不加axis=1
        return np.exp(-(np.linalg.norm(x) ** 2) / (2 * (sigma ** 2)))
    else:  # 当x为一个矩阵时，按行求范式
        return np.exp(-(np.linalg.norm(x, axis=1) ** 2) / (2 * (sigma ** 2)))


class SVM():

    def __init__(self):
        self.b = 0
        self.alpha = []
        self.C = 1  # 正则项系数，默认1
        self.sigma = 1  # 高斯核函数参数，默认1

    def train(self, data_train):
        # 获取数据总数
        N = np.size(data_train, 0)
        # 计算二次型参数
        P = np.zeros((N, N))
        for n in range(0, N):
            for m in range(0, N):
                P[m, n] = data_train[m, 2] * data_train[n, 2] * kernel_gauss(data_train[m, :2] - data_train[n, :2],
                                                                             self.sigma)
        P = matrix(P)
        q = matrix(-1 * np.ones(N))
        G = matrix(np.append(-1 * np.eye(N), np.eye(N), axis=0))
        h = matrix(np.append(np.zeros(N), self.C * np.ones(N)))
        A = matrix(np.array(data_train[:, 2]), (1, N))
        b = matrix(0, tc='d')
        # 计算解
        sol = solvers.qp(P, q, G, h, A, b)
        self.alpha = sol['x']
        for i in range(0, N):
            if self.alpha[i] > 1.0e-4 and self.alpha[i] < self.C - 1.0e-4:
                self.b = data_train[i, 2] - np.sum(
                    kernel_gauss(data_train[:, :2] - data_train[i, :2], self.sigma) * data_train[:, 2] * np.transpose(
                        self.alpha))
                break

    def predict(self, x):
        result = np.zeros(np.shape(x)[0])
        for i in range(0, np.shape(x)[0]):
            result[i] = np.sum(kernel_gauss(data_train[:, :2] - x[i, :], self.sigma) * data_train[:, 2] * np.transpose(
                self.alpha)) + self.b
        return np.where(result > 0, 1, -1)


if __name__ == '__main__':
    # Load dataset
    train_file = 'train_kernel.txt'
    test_file = 'test_kernel.txt'
    data_train = load_data(train_file)  # dataset format [x1, x2, t], shape (N * 3)
    data_test = load_data(test_file)

    # train SVM
    svm = SVM()
    svm.train(data_train)

    # predict
    x_train = data_train[:, :2]  # features [x1, x2]
    t_train = data_train[:, 2]  # ground truth labels
    t_train_pred = svm.predict(x_train)  # predicted labels
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # evaluate
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))

    show_data(data_test)
