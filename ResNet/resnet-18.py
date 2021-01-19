import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -*-*- 定义残差块 -*-*-
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)

    def forward(self,X):

        y = self.conv1(X)
        Y = F.relu(self.BN1(self.conv1(X)))
        Y = self.BN2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y+X)

# -*-*- 查看残差块 -*-*-
# -*- 查看输入和输出形状一致的情况 -*-
blk = Residual(3,3)
X = torch.rand((4,3,6,6))
print(blk(X).shape)       # 输出形状与输入形状一致

# -*- 增加输出通道数的同时减半输出的高和宽 -*-
blk = Residual(3, 6, use_1x1conv=True, stride=2)
print(blk(X).shape)


# -*-*- 逐渐组建网络模型(用add_module()方法) -*-*-
# -*- 开头两层分别是7x7的卷积层和3x3的最大池化层 -*-
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    """
    :param in_channels:
    :param out_channels:
    :param num_residuals: 本模块里包含的残差块数目
    :param first_block: 判断本次是否构建的是第一个模块
    :return:
    """
    if first_block:
        assert in_channels == out_channels   # 第一个模块的通道数与输入通道数一致
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 除了第一个模块是输入通道数与输出通道数一样之外，其他模块都不一样，所以要用1x1卷积块, 且其他模块要使高宽减半，要使stride为2
            block.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            # 即当前构建的是第一个模块（第一个模块的通道数与输入通道数一致，不需要1x1卷积层，也不用使高宽减半）
            # 或者当前构建的是非第一个模块的非第一个残差块（第一个残差块已经用了高宽减半操作stride=2，也指明了要用1x1卷积处理输入，这里就不用指明了）
            block.append(Residual(out_channels, out_channels))
    return nn.Sequential(*block)      # 为啥要加*？？


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self,x):
        return x.view(x.shape[0], -1)


# -*- 为网络加入所有残差块(这里每个模块有两个残差块)，加入全局平均池化层，加上全连接层 -*-
net.add_module("resnet_block1", resnet_block(64,64,2,first_block=True))
net.add_module("resnet_block2", resnet_block(64,128,2))
net.add_module("resnet_block3", resnet_block(128,256,2))
net.add_module("resnet_block4", resnet_block(256,512,2))
net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))  # 输出shape: [batch, 512, 1, 1]
net.add_module("fc", nn.Sequential(
    FlattenLayer(),
    nn.Linear(512, 10)
))

# -*- 观察在不同模块之间的输入shape的变化 -*-
X = torch.rand((1,1,224,224))
for name, BLK in net.named_children():
    X = BLK(X)
    print(f"name:{name}, output shape:{X.shape}")


def load_FashionMNIST_data(batch_size=64, size=None, root="../softmax/DatasetsTemp"):
    """
    读取数据
    :param batch_size:
    :param size:
    :param root:
    :return:
    """
    trans = []
    if size:
        trans.append(torchvision.transforms.Resize(size=size))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)
    train_data = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
    test_data = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)
    train_iter = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_iter = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter


def evaluate_acc(test_iter, net):
    test_n, test_correct_n = 0, 0
    for x, y in test_iter:
        output = net(x)
        test_correct_n = (output.argmax(dim=1) == y).sum()
        test_n += y.shape[0]
    test_acc = round(test_correct_n / test_n, 3)
    return test_acc


if __name__ == '__main__':
    epochs = 5
    batch_size = 64
    lr = 0.0001
    new_img_size = 96

    # -*-*- 损失函数 -*-*-
    loss = nn.CrossEntropyLoss()

    # -*-*- 优化器 -*-*-
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_iter, test_iter = load_FashionMNIST_data(batch_size=batch_size, size=new_img_size)

    net = net.to(device)
    # -*-*- 训练 -*-*-
    print(f"training on {device}")
    for epoch in range(epochs):
        train_loss_sum, train_n, train_correct_n = 0, 0, 0
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            l = loss(output, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_correct_n += (output.argmax(dim=1) == y).sum()
            train_n += y.shape[0]
        train_avg_loss = round(train_loss_sum / train_n, 4)
        train_acc = round(train_correct_n / train_n, 3)
        test_acc = evaluate_acc(test_iter, net)
        print(f"epoch:{epoch}, train avg loss:{train_avg_loss}, train acc:{train_acc}, test acc:{test_acc}")
