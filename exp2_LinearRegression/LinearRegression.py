import numpy as np
import matplotlib.pyplot as plt
import csv

index = "ROA(C) before interest and depreciation before interest"
Num = 200  # 选取数据范围[1-6800]


# 读取数据文件
def loadDataset(file_name):
    with open(file_name, "r") as csvfile:
        reader = csv.reader(csvfile)
        column = [row[1] for row in reader]
    column = column[1:]
    column = [i for i in column if i != "0"]
    column = column[:Num]
    y = range(1, len(column) + 1, 1)
    data = []
    for i in column:
        data.append(float(i))
    label = []
    for j in y:
        label.append(float(j))
    return data, label


# 最小二乘法
def leastSquare(X, Y):
    X = np.mat(X)
    Y = np.mat(Y)
    W = (X * X.T).I * X * Y.T  # 转置.T  逆运算.I
    return W


if __name__ == "__main__":
    # 构建数据集
    data, times = loadDataset("data.csv")
    x1 = np.ones(Num)
    # 最小二乘法
    W = leastSquare([x1, times], data)

    # 计算loss
    mae = 0
    mse = 0
    for i in range(Num):
        mae += abs(W[1, 0] * times[i] + W[0, 0] - data[i])
        mse += (W[1, 0] * times[i] + W[0, 0] - data[i])**2
    print('平均绝对误差: ', mae/ Num)
    print('均方误差: ', mse/Num)

    # 画图
    plt.scatter(times, data, color="gray", s=10)
    plt.xlabel("Time", fontsize=10)  # 横坐标
    plt.ylabel("ROA(C)", fontsize=10)  # 纵坐标
    plt.title(index + "(partial)")  # 标题
    # plt.ylim(0, 1)
    x1 = np.linspace(0, Num, 5)  # 拟合直线x范围
    y1 = W[1, 0] * x1 + W[0, 0]  # 拟合直线纵坐标取值
    plt.plot(x1, y1, color="red", linewidth=2, linestyle=':')
    plt.figtext(0.7,0.14, 'MSE: {:.7f}'.format(mse/Num))
    plt.show()
