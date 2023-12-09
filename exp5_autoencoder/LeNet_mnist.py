import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data
import csv


class Net(nn.Module):
    def __init__(self):
        # 复制并使用Net的父类的初始化方法，即nn.Module的初始化函数
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)  # 卷积层1,输入是单通道灰度图
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)  # 卷积层2
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层1
        self.fc2 = nn.Linear(120, 84)  # 全连接层2
        self.fc3 = nn.Linear(84, 10)  # 全连接层3

    def forward(self, x):
        x = self.conv1(x)  # 32*32->28*28
        x = F.relu(x)
        x = self.pool(x)  # 28*28->14*14
        x = self.conv2(x)  # 14*14->10*10
        x = F.relu(x)
        x = self.pool(x)  # 10*10-> 5*5
        x = x.view(x.size(0), -1)
        x = self.fc1(x)  # 16*5*5->120
        x = F.relu(x)
        x = self.fc2(x)  # 120->84
        x = F.relu(x)
        x = self.fc3(x)  # 84->10
        return x


def writeCsv(data):
    header = ['epoch', 'train_loss', 'val_loss', 'val_acc']
    with open(f'benchmark_LeNet.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


BATCH_SIZE = 64
EPOCH = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)

net = Net().to(device)  # 实例化网络
loss_fuc = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 导入数据集, 进行预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)


CSVdata = []
for epoch in range(EPOCH):
    sum_loss = 0
    # train
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 传递损失,更新参数
        output = net(inputs)
        loss = loss_fuc(output, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    print(f'Epoch: {epoch + 1:d}\ntrain loss: {sum_loss / len(train_loader):.03f}')

    correct = 0
    total = 0
    val_loss = 0
    for data in test_loader:
        test_inputs, labels = data
        test_inputs, labels = test_inputs.to(device), labels.to(device)
        outputs_test = net(test_inputs)
        val_loss += loss_fuc(outputs_test, labels).item()
        _, predicted = torch.max(outputs_test.data, 1)  # 相当于np.argmax()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'test loss: {val_loss / len(test_loader):.3f}')
    print(f'test accuracy：{100 * correct / total:.3f}%')
    CSVdata.append([epoch, sum_loss / len(train_loader), val_loss / len(test_loader), correct / total])
writeCsv(CSVdata)
