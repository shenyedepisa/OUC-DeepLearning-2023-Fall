import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

path = './iris/iris'
test_ratio = 0.3


def readData(dirPath):
    data = pd.read_csv(dirPath)  # 读数据集
    setTypes = list(set(data['class'].values))
    num = [i for i in range(len(setTypes))]
    types = dict((i, j) for i, j in zip(setTypes, num))
    data = data.sample(frac=1, random_state=0)  # 不随机打乱
    length = len(data)
    petalData = data[['petal length', 'petal width']].values
    txtLabels = data['class'].values
    labels = [types[i] for i in txtLabels]
    # 划分训练集和测试集
    train_data = np.array(petalData[int(test_ratio * length):])
    train_target = np.array(labels[int(test_ratio * length):])
    test_data = np.array(petalData[:int(test_ratio * length)])
    test_target = np.array(labels[:int(test_ratio * length)])

    return train_data, train_target, test_data, test_target


def writeCsv(data):
    header = ['epoch', 'train_loss', 'val_loss', 'val_acc']
    with open(f'benchmark.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


class MLP:
    # initialise the neural network
    def __init__(self, input_dim, hidden_layer1_dim, hidden_layer2_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_layer1_dim = hidden_layer1_dim
        self.hidden_layer2_dim = hidden_layer2_dim
        self.output_dim = output_dim

        # 正态分布初始化权重
        self.weights1 = np.random.randn(input_dim, hidden_layer1_dim)
        self.weights2 = np.random.randn(hidden_layer1_dim, hidden_layer2_dim)
        self.weights3 = np.random.randn(hidden_layer2_dim, output_dim)
        self.b1 = np.random.randn(hidden_layer1_dim)
        self.b2 = np.random.randn(hidden_layer2_dim)
        self.b3 = np.random.randn(output_dim)

    # train the network using training data set
    def forward(self, inputs):
        self.x = np.array(inputs, ndmin=2)  # 调整通道数
        self.z1 = np.dot(self.x, self.weights1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.b2
        self.a2 = np.tanh(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.b3
        self.z3 = self.softmax(self.z3)
        return self.z3

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def getLoss(self, y, output):
        targets = np.zeros(3)
        targets[y] = 1
        return output - targets

    def tanh_derivative(self, x):
        return 1 - np.power(x, 2)

    def backward(self, y, output, learning_rate=0.01):
        self.loss = self.getLoss(y, output)
        d_z3 = self.loss
        d_W3 = np.dot(self.a2.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0)

        d_a2 = np.dot(d_z3, self.weights3.T)
        d_z2 = d_a2 * self.tanh_derivative(self.a2)
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0)

        d_a1 = np.dot(d_z2, self.weights2.T)
        d_z1 = d_a1 * self.tanh_derivative(self.a1)
        d_W1 = np.dot(self.x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)

        self.weights1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.weights2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        self.weights3 -= learning_rate * d_W3
        self.b3 -= learning_rate * d_b3

        return self.loss


def draw(x, y, name):
    setosa = np.array([[x[i][0], x[i][1]] for i in range(len(y)) if y[i] == 0])
    versicolor = np.array([[x[i][0], x[i][1]] for i in range(len(y)) if y[i] == 1])
    virginica = np.array([[x[i][0], x[i][1]] for i in range(len(y)) if y[i] == 2])

    plt.figure()
    plt.title(name, fontsize=12)
    plt.xlabel("petal length (cm)", fontsize=10)  # 横坐标
    plt.ylabel("petal width (cm)", fontsize=10)  # 纵坐标
    if len(setosa > 0):
        plt.scatter(setosa[:, 0], setosa[:, 1], s=12, c='r', marker='o', label='setosa')
    if len(versicolor > 0):
        plt.scatter(versicolor[:, 0], versicolor[:, 1], s=12, c='g', marker='o', label='versicolor')
    if len(virginica > 0):
        plt.scatter(virginica[:, 0], virginica[:, 1], s=12, c='b', marker='o', label='virginica')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    train_x, train_y, test_x, test_y = readData(path)
    model = MLP(2, 20, 10, 3)
    EPOCH = 200  # 迭代次数
    CSVdata = []
    for epoch in range(EPOCH):
        sum_loss = 0
        # 数据读取
        train_num = len(train_y)
        for i, data in enumerate(zip(train_x, train_y)):
            inputs, labels = data
            output = model.forward(inputs)
            loss = model.backward(labels, output, 0.01)
            sum_loss += np.sum(abs(loss))
            # 每训练100个batch打印一次平均loss

        print(f'Epoch: {epoch + 1:d}\ntrain loss: {sum_loss / train_num:.03f}')

        correct = 0
        val_loss = 0
        val_num = len(test_y)
        pred_labels = []
        for i, data in enumerate(zip(test_x, test_y)):
            test_inputs, labels = data
            output = model.forward(test_inputs)
            loss = model.getLoss(labels, output)
            val_loss += np.sum(abs(loss))
            predicted = np.argmax(output)
            pred_labels.append(predicted)
            correct += (predicted == labels)
        if epoch % 99 == 0:
            draw(test_x, pred_labels, 'iris classification(MLP: prediction)')
        print(f'val loss: {val_loss / val_num:.03f}')
        print(f'第{epoch + 1}个epoch的识别准确率为：{100 * correct / val_num:.03f}%')
        CSVdata.append([epoch, sum_loss / train_num, val_loss / val_num, correct / val_num])
    writeCsv(CSVdata)
