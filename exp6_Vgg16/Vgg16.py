import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

batch_size = 64
EPOCH = 50
learningRate = 0.01
# 加载数据
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
train_dataset = torchvision.datasets.CIFAR10(
    root="./data/", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data/", train=False, transform=transform, download=True
)
# 划分训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(123)
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.Conv1 = nn.Sequential(
            # CIFAR10 数据集是彩色图 - RGB三通道, 所以输入通道为 3, 图片大小为 32*32
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            # inplace-选择是否对上层传下来的tensor进行覆盖运算, 可以有效地节省内存/显存
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.Conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.Conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


model = VGG16().to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=0.8, weight_decay=0.001)
# 动态学习率
schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.6, last_epoch=-1)

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []


# train
def train(epoch):
    for epoch in range(epoch):
        running_loss = 0.0
        correct = 0.0
        total = 0
        model.train()
        train_tqdm = tqdm(
            enumerate(train_loader, 0),
            total=len(train_loader),
            ncols=100,
            mininterval=1,
        )
        for i, (inputs, labels) in train_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            total += inputs.size(0)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
            loss = criterion(outputs, labels).to(device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += torch.eq(pred, labels).sum().item()
            running_loss += loss.item()

        train_tqdm.close()
        train_loss_list.append(running_loss / len(train_loader))
        train_acc_list.append(correct / total)

        model.eval()
        val_correct = 0.0
        val_total = 0
        val_loss = 0.0
        val_tqdm = tqdm(
            enumerate(val_loader, 0),
            total=len(val_loader),
            ncols=100,
            mininterval=1,
        )
        for i, (inputs, labels) in val_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            val_total += inputs.size(0)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
            loss = criterion(outputs, labels).to(device)
            val_loss += loss.item()
            val_correct += torch.eq(pred, labels).sum().item()

        val_loss_list.append(val_loss / len(val_loader))
        val_acc_list.append(val_correct / val_total)
        val_tqdm.close()
        print(f"epoch = {epoch + 1}, train loss = {running_loss / len(train_loader):.6f}")
        print(f"epoch = {epoch + 1}, train Accuracy:{(100 * correct / total):.3f}%")
        print(f"epoch = {epoch + 1}, val loss = {val_loss / len(val_loader):.6f}")
        print(
            f"epoch = {epoch + 1}, val Accuracy:{(100 * val_correct / val_total):.3f}%"
        )

        # 每一轮结束输出一下当前的学习率 lr
        lr_1 = opt.param_groups[0]["lr"]
        print(f"learn_rate:{lr_1:.6f}")
        schedule.step()



    # 训练过程可视化
    x = np.arange(1, EPOCH + 1)
    plt.plot(x, train_loss_list)
    plt.plot(x, val_loss_list)
    plt.title("Loss")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.legend(["train loss", "val loss"])
    plt.savefig("./loss_img.png")
    plt.show()


    plt.plot(x, train_acc_list)
    plt.plot(x, val_acc_list)
    plt.title("Accuracy")
    plt.ylabel("acc")
    plt.xlabel("Epoch")
    plt.legend(["train acc", "val acc"])
    plt.savefig("./acc_img.png")
    plt.show()


# Test
def test():
    model.eval()
    correct = 0.0
    total = 0
    # 训练模式不需要反向传播更新梯度
    with torch.no_grad():
        for _, (inputs, labels) in tqdm(
            enumerate(test_loader, 0),
            total=len(test_loader),
            ncols=100,
            mininterval=1,
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
            total += inputs.size(0)
            correct += torch.eq(pred, labels).sum().item()

    print(f"test Accuracy:{(100 * correct / total):.3f}%")


if __name__ == "__main__":
    start = time.time()
    train(EPOCH)
    test()
    end = time.time()
    print(f"time:{end - start:.3f}s")
