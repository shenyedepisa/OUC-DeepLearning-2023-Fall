# OUC-DeepLearning-2023-Fall

中国海洋大学深度学习秋季课程

仅供参考, 请勿抄袭

理论上会在DDL之后一天更新

(咕咕)

### exp1 perceptron

一个简单的感知机, 没有任何优化, 

对于线性可分问题, 迭代足够多是一定可以分开的,

对于不可分问题, 准确率是震荡的, 比较随缘

### exp2 LinearRegression

数据源: [台湾经济数据1999-2009](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)

本次实验对“息前资产收益率和息前折旧”条目进行线性回归

### exp3 KNN

数据集为150条鸢尾花数据, 三种鸢尾花各50条, [iris](https://archive.ics.uci.edu/dataset/53/iris)

使用KNN算法根据花瓣的长和宽对三种鸢尾花进行分类，分类准确率为 97.8%

### exp4 mlp & bp

数据集同exp3 [iris](https://archive.ics.uci.edu/dataset/53/iris)

用numpy实现一个包含两个隐藏层的mlp模型,并实现反向传播, 分类准确率为 97.8%

(但是实际收敛情况不如KNN稳定)

### exp5 mlp & autoencoder & LeNet

手写数字识别分类, 数据集: [MNIST](http://yann.lecun.com/exdb/mnist/)

实现了三个模型:

包含四个隐藏层的mlp, 准确率 98%

包含四个隐藏层的自编码器 + 分类头, 准确率 89%

LeNet-5, 准确率 98%

### exp6 Vgg16

一个经典的Vgg16模型, 在cifar10数据集上分类准确率87%


