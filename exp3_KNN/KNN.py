import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

path = './iris/iris'
test_ratio = 0.3


def readData(dirPath):
    data = pd.read_csv(dirPath)  # 读数据集
    setTypes = list(set(data['class'].values))
    num = [i for i in range(len(setTypes))]
    types = dict((i, j) for i, j in zip(setTypes, num))
    data = data.sample(frac=1, random_state=0)  # 随机打乱
    length = len(data)
    petalData = data[['petal length', 'petal width']].values
    txtLabels = data['class'].values
    labels = [types[i] for i in txtLabels]
    # 划分训练集和测试集
    train_data = np.array(petalData[int(test_ratio * length):])
    train_type = np.array(labels[int(test_ratio * length):])
    test_data = np.array(petalData[:int(test_ratio * length)])
    test_type = np.array(labels[:int(test_ratio * length)])

    return train_data, train_type, test_data, test_type


def knn(train_data, train_labels, test_data, k):
    distances = np.sqrt(np.sum((train_data - test_data) ** 2, axis=1))  # 计算待测点与已知点的距离
    k_indices = np.argsort(distances)[:k]  # 取最近的k个点
    k_nearest_labels = train_labels[k_indices]  # 取这k个点的类别
    most_common = Counter(k_nearest_labels.ravel()).most_common(1)  # 统计出现最多的类别[类别,出现次数]
    return most_common[0][0]


def draw(x, y, name):
    train_setosa = np.array([[x[i][0], x[i][1]] for i in range(len(y)) if y[i] == 0])
    train_versicolor = np.array([[x[i][0], x[i][1]] for i in range(len(y)) if y[i] == 1])
    train_virginica = np.array([[x[i][0], x[i][1]] for i in range(len(y)) if y[i] == 2])

    plt.figure()
    plt.title(name, fontsize=12)
    plt.xlabel("petal length (cm)", fontsize=10)  # 横坐标
    plt.ylabel("petal width (cm)", fontsize=10)  # 纵坐标
    plt.scatter(train_setosa[:, 0], train_setosa[:, 1], s=12, c='r', marker='o', label='setosa')
    plt.scatter(train_versicolor[:, 0], train_versicolor[:, 1], s=12, c='g', marker='o', label='versicolor')
    plt.scatter(train_virginica[:, 0], train_virginica[:, 1], s=12, c='b', marker='o', label='virginica')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = readData(path)
    pred_labels = []
    for data in test_x:
        pred_labels.append(knn(train_x, train_y, data, k=5))

    # 计算准确率
    n = np.sum(pred_labels != test_y.ravel())
    accuracy = np.sum(pred_labels == test_y.ravel()) / len(test_y)
    print('测试集聚类的准确率为: {:.3f}%\n聚类错误的点个数: {}'.format(accuracy*100, n))

    # 画图
    draw(train_x, train_y, 'iris classification(train)')
    draw(test_x, test_y, 'iris classification(test)')
    draw(test_x, pred_labels, 'iris classification(KNN: prediction)')
