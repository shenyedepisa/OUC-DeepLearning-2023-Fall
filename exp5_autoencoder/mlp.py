import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils import data
import csv


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mlp_classifier = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.mlp_classifier(x)
        return x


def writeCsv(data):
    header = ['epoch', 'train_cls_loss', 'test_cls_loss', 'test_acc']
    with open(f'benchmark_MLP.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


BATCH_SIZE = 64
EPOCH = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)

net = Net().to(device)  # 实例化网络
loss_fuc = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
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
        inputs = inputs.view(inputs.size(0), -1).to(device)
        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 传递损失,更新参数
        output = net(inputs)
        loss = loss_fuc(output, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    print(f'Epoch: {epoch + 1:d}\ntrain loss: {sum_loss / len(train_loader):.5f}')

    correct = 0
    total = 0
    val_loss = 0
    for data in test_loader:
        test_inputs, labels = data
        test_inputs = test_inputs.view(test_inputs.size(0), -1).to(device)
        test_inputs, labels = test_inputs.to(device), labels.to(device)
        outputs_test = net(test_inputs)
        val_loss += loss_fuc(outputs_test, labels).item()
        _, predicted = torch.max(outputs_test.data, 1)  # 相当于np.argmax()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'test loss: {val_loss / len(test_loader):.5f}')
    print(f'test accuracy：{100 * correct / total:.3f}%')

    CSVdata.append([epoch, sum_loss / len(train_loader), val_loss / len(test_loader), correct / total])

writeCsv(CSVdata)
