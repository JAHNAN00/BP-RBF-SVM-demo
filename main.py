import numpy as np
import random
import matplotlib.pyplot as plt

from src.bp import BP
from src.rbf import RBF
from src.svm import SVM

def task_bp():
    
    # 任务一 拟合函数
    # 函数一：y=1/x**5
    x1 = np.linspace(1, 10, 1000).reshape(-1,1)
    np.random.shuffle(x1)
    y1 = 1/x1**5
    noise = np.random.normal(0, 0.01, y1.shape)  # 均值为 0，标准差为 0.01 的高斯噪声
    y1 = y1 + noise

    # 函数二：y=(1+cosx)/2
    x2 = np.linspace(0, 6*np.pi, 1000).reshape(-1,1)
    np.random.shuffle(x2)
    y2=(1+np.cos(x2))/2
    noise = np.random.normal(0, 0.01, y2.shape)  # 均值为 0，标准差为 0.01 的高斯噪声
    y2= y2+noise
    
    # 函数三：z=1/(sqrt(x^2+y^2))
    x3 = np.random.uniform(-20, 20, 1000)
    y3 = np.random.uniform(-20, 20, 1000)
    input3 = np.column_stack((x3,y3))
    z3=1/np.sqrt(x3**2+y3**2)
    noise = np.random.normal(0, 0.01, z3.shape)  # 均值为 0，标准差为 0.01 的高斯噪声
    # z3 = z3 + noise

    # bp网络拟合
    bp1 = BP([1, 25, 1], activation_function='sigmoid')
    bp1.train(x1, y1, epochs=20000, learning_rate=0.3)
    y1_predict = bp1.predict(x1)

    bp2 = BP([1, 25, 1], activation_function='sigmoid')
    bp2.train(x2, y2, epochs=100, learning_rate=0.3)
    y2_predict = bp2.predict(x2)

    bp3 = BP([2, 25, 1], activation_function='sigmoid')
    bp3.train(input3, z3, epochs=20000, learning_rate=0.3)
    z3_predict= bp3.predict(input3)

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.subplot(1,2,1)
    plt.scatter(x1, y1, label='Original')
    plt.scatter(x1, y1_predict, label='Predict')
    plt.legend()
    plt.subplot(1,2,2)
    plt.scatter(x2, y2, label='Original')
    plt.scatter(x2, y2_predict, label='Predict')
    plt.legend()
    plt.savefig('task1_bp_2d.png')

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    scatter = ax.scatter(x3, y3, z3 ,c=x3, cmap='viridis')
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Z value')
    ax = fig.add_subplot(122, projection='3d')
    scatter = ax.scatter(x3, y3, z3_predict ,c=x3, cmap='viridis')
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Z value')
    plt.savefig('task1_bp_3d.png')


def task_rbf():
    # 任务一 拟合函数
    # 函数一：y=1/x**5
    x1 = np.linspace(1, 10, 1000).reshape(-1,1)
    np.random.shuffle(x1)
    y1 = 1/x1**5
    noise = np.random.normal(0, 0.01, y1.shape)  # 均值为 0，标准差为 0.01 的高斯噪声
    y1 = y1 + noise

    # 函数二：y=(1+cosx)/2
    x2 = np.linspace(0, 6*np.pi, 1000).reshape(-1,1)
    np.random.shuffle(x2)
    y2=(1+np.cos(x2))/2
    noise = np.random.normal(0, 0.01, y2.shape)  # 均值为 0，标准差为 0.01 的高斯噪声
    y2= y2+noise
    
    # 函数三：z=1/(sqrt(x^2+y^2))
    x3 = np.random.uniform(-20, 20, 1000)
    y3 = np.random.uniform(-20, 20, 1000)
    input3 = np.column_stack((x3,y3))
    z3=1/np.sqrt(x3**2+y3**2)
    noise = np.random.normal(0, 0.01, z3.shape)  # 均值为 0，标准差为 0.01 的高斯噪声
    # z3 = z3 + noise


    # rbf拟合
    rbf1 = RBF(kernel='gaussian', gamma=0.1)
    rbf1.fit(x1, y1, num_centers=10)
    y1_predict = rbf1.predict(x1)

    rbf2 = RBF(kernel='gaussian', gamma=0.1)
    rbf2.fit(x2, y2, num_centers=10)
    y2_predict = rbf2.predict(x2)

    rbf3 = RBF(kernel='gaussian', gamma=0.1)
    rbf3.fit(input3, z3, num_centers=10)
    z3_predict = rbf3.predict(input3)

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.subplot(1,2,1)
    plt.scatter(x1, y1, label='Original')
    plt.scatter(x1, y1_predict, label='Predict')
    plt.legend()
    plt.subplot(1,2,2)
    plt.scatter(x2, y2, label='Original')
    plt.scatter(x2, y2_predict, label='Predict')
    plt.legend()
    plt.savefig('task1_rbf_2d.png')

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    scatter = ax.scatter(x3, y3, z3 ,c=x3, cmap='viridis')
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Z value')
    ax = fig.add_subplot(122, projection='3d')
    scatter = ax.scatter(x3, y3, z3_predict ,c=x3, cmap='viridis')
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Z value')
    plt.savefig('task1_rbf_3d.png')

def task_svm():
    # 任务一 拟合函数
    x = np.linspace(1e-10, 10, 30)
    y = np.linspace(1e-10, 1, 30)

    # 使用meshgrid生成点阵
    x_grid, y_grid = np.meshgrid(x, y)

    # 将点阵转换为一维数组，每个点为(x, y)的形式
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    labels1 = [-1 if y > (1+np.cos(x))/2+0.2 else 1 for x,y in points]
    labels2 = [-1 if y > (1+np.cos(x))/2-0.2 else 1 for x,y in points]
    labels= [a * b for a, b in zip(labels1, labels2)]
    colors = ['red' if label==-1 else 'blue' for label in labels]
    plt.scatter(points[:,0],points[:,1],c=colors)
    plt.savefig('task1_svm_2d.png')

    svm1 = SVM()
    svm2 = SVM()
    data_train1=np.column_stack((points, labels1))
    svm1.train(data_train1)
    pred1=svm1.predict(data_train1)
    data_train2=np.column_stack((points, labels2))
    svm2.train(data_train2)
    pred2=svm2.predict(data_train2)
    pred=[a * b for a, b in zip(pred1, pred2)]
    def show_data(data):
        fig, ax = plt.subplots()
        cls = data[:, 2]
        ax.scatter(data[:, 0][cls == 1], data[:, 1][cls == 1])
        ax.scatter(data[:, 0][cls == -1], data[:, 1][cls == -1])
        ax.grid(False)
        fig.tight_layout()
        plt.savefig('task1_svm_2d.png')
    show_data(pred)



if __name__ == "__main__":
   np.random.seed(0)
   #task_bp()
   #task_rbf()
   task_svm()