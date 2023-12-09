import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils import data
import csv

BATCH_SIZE = 64
EPOCH = 50
# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)


def writeCsv(data):
    header = ['epoch', 'train_L1_loss', 'train_L2_loss', 'train_cls_loss', 'test_l1_loss', 'test_l2_loss',
              'test_cls_loss', 'train_total_loss', 'test_total_loss', 'test_acc']
    with open(f'benchmark_SAE1.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


# 定义堆叠自编码器模型
class StackedAutoencoder(nn.Module):
    def __init__(self):
        super(StackedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 20)
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(20, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


# 实例化模型并定义损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
SAEmodel = StackedAutoencoder().to(device)
criterion1 = nn.L1Loss()
criterion2 = nn.MSELoss()
saeOptimizer = torch.optim.Adam(SAEmodel.parameters(), lr=0.001)

classifierHead = classifier().to(device)  # 实例化网络
loss_fuc = nn.CrossEntropyLoss()  # 交叉熵损失函数
headOptimizer = torch.optim.Adam(classifierHead.parameters(), lr=0.001)

# 训练模型
CSVdata = []
for epoch in range(EPOCH):
    L1_loss = 0
    L2_loss = 0
    cls_loss = 0
    SAEmodel.train()
    classifierHead.train()
    for img, labels in train_loader:
        img = img.view(img.size(0), -1).to(device)
        labels = labels.to(device)

        # 训练SAE
        saeOptimizer.zero_grad()
        encodeImg, decodeImg = SAEmodel(img)
        loss1 = criterion1(decodeImg, img)
        loss2 = criterion2(decodeImg, img)
        L1_loss += loss1.item()
        L2_loss += loss2.item()
        loss1.backward()
        saeOptimizer.step()

        # 训练分类头
        headOptimizer.zero_grad()
        outputs = classifierHead(encodeImg.detach())
        loss2 = loss_fuc(outputs, labels)
        cls_loss += loss2.item()
        loss2.backward()
        headOptimizer.step()

    print(
        f'Epoch: {epoch + 1:d}\n'
        f'train L1 loss: {L1_loss / len(train_loader):.5f}, '
        f'L2 loss: {L2_loss / len(train_loader):.5f}, '
        f'cls loss:{cls_loss / len(train_loader):.5f}, '
        f'total loss: {(cls_loss + L1_loss + L2_loss) / len(train_loader):.5f}'
    )

    # 测试模型
    SAEmodel.eval()
    classifierHead.eval()
    correct = 0
    total = 0
    test_L1_loss = 0
    test_L2_loss = 0
    test_cls_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            encodeImg, decodeImg = SAEmodel(images)
            loss1 = criterion1(decodeImg, images)
            loss2 = criterion2(decodeImg, images)
            test_L1_loss += loss1.item()
            test_L2_loss += loss2.item()

            outputs = classifierHead(encodeImg)
            loss2 = loss_fuc(outputs, labels)
            test_cls_loss += loss2.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        f'test L1 loss: {test_L1_loss / len(test_loader):.5f}, '
        f'L2 loss: {test_L2_loss / len(test_loader):.5f}, '
        f'cls loss:{test_cls_loss / len(test_loader):.5f}, '
        f'total loss: {(test_cls_loss + test_L1_loss + test_L2_loss) / len(test_loader):.5f}\n'
        f'test accuracy：{100 * correct / total:.3f}%'
    )

    CSVdata.append([epoch, L1_loss / len(train_loader), L2_loss / len(train_loader), cls_loss / len(train_loader),
                    test_L1_loss / len(test_loader), test_L2_loss / len(test_loader), test_cls_loss / len(test_loader),
                    (cls_loss + L1_loss + L2_loss) / len(train_loader),
                    (test_cls_loss + test_L1_loss + test_L2_loss) / len(test_loader), correct / total])

writeCsv(CSVdata)
