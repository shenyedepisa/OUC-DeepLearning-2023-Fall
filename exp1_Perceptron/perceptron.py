import numpy as np
import matplotlib.pyplot as plt


# 生成数组
dataX0 = np.random.randn(30, 2)  # 正样本
targetX0 = np.ones(30)
dataX1 = np.random.randn(30, 2)  # 负样本
targetX1 = np.zeros(30)
dataX0 += 2
dataX1 -= 2
# 对原始数据可视化
dataset = np.concatenate([dataX0, dataX1], axis=0)
target = np.hstack((targetX0, targetX1))
plt.scatter(dataset[:, 0], dataset[:, 1])
plt.title("original data")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.scatter(dataset[:30, 0], dataset[:30, 1], color="red")
plt.scatter(dataset[30:, 0], dataset[30:, 1], color="blue")
plt.show()


# 感知机方法, x是数据坐标, y是标签, 返回w,b和推理结果f
def perceptron(x, y):
    loss = 1
    n = 0
    weight = np.random.random(x.shape[1])
    b = np.random.random(1)
    f = np.zeros(x.shape[0])
    newPlot(weight, b, dataset, 0, "initial")
    while loss != 0 and n < 5:
        for i in range(x.shape[0]):
            p = np.dot(weight, x[i]) + b
            f[i] = 1 if p > 0 else 0
            if y[i] - f[i] != 0:  # 结果不一致时更新权重
                weight = weight + (y[i] - f[i]) * x[i]
                b = b + (y[i] - f[i])
                newPlot(weight, b, dataset, i + 1, "update")  # 权重更新时绘图
        n += 1
        loss = np.sum(np.abs(y - f))
    print("weight: ", weight)
    print("bias:", b)
    return weight, b, f


# 使用训练好的感知机推理, 返回分出的两类数据
def infer(weight, bias, data, n):
    posData = []
    negData = []
    f = np.zeros(n)
    for i in range(n):
        p = np.dot(weight, data[i]) + bias
        f[i] = 1 if p > 0 else 0
        if f[i] == 1:
            posData.append(data[i])
        else:
            negData.append(data[i])
    return np.array(posData), np.array(negData)


# 绘图,画分割线用
def getLine(weight, bias, lineData):
    output = -(weight[0] * lineData + bias) / weight[1]
    return output


# 绘图
def newPlot(weight, bias, data, n, name):
    lineX = np.linspace(-5, 5, 2)
    lineY = getLine(weight, bias, lineX)
    pos, neg = infer(weight, bias, data, n)
    plt.plot(lineX, lineY, "k")
    plt.scatter(dataset[:, 0], dataset[:, 1], color="gray")
    if pos.shape[0] != 0:
        plt.scatter(pos[:, 0], pos[:, 1], color="red")
    if neg.shape[0] != 0:
        plt.scatter(neg[:, 0], neg[:, 1], color="blue")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(name)
    plt.show()


if __name__ == "__main__":
    W, B, yOut = perceptron(dataset, target)
    newPlot(W, B, dataset, yOut.shape[0], "final result")
    sumNum = 0
    for i in range(yOut.shape[0]):
        if yOut[i] == target[i]:
            sumNum += 1
    print("acc: ", sumNum / yOut.shape[0])
