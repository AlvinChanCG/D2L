import torch
import torchvision
import torch.utils.data as Data
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -*-*- 定义dense block里的卷积块(这里所有卷积块的输出channel数都是一样的) -*-*-
def conv_block(in_channel, out_channel):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel,kernel_size=3, padding=1)
    )
    return blk

# -*-*- 构建稠密块 -*-*-
class Denseblock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(Denseblock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i*out_channels     # 构建第i+1个卷积块时，其输入channel数
            net.append(conv_block(in_channel=in_c, out_channel=out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs*out_channels    # 整个Denseblock的输出通道数；用于后面逐渐堆叠dense block时调用

    def forward(self,x):
        for BLK in self.net:
            y = BLK(x)
            x = torch.cat((x,y), dim=1)  # 在通道维上将输入输出连接
        return x


# -*-*- 定义过渡层 -*-*-
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels,out_channels,kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self,x):
        return x.view(x.shape[0], -1)


# -*-*- 定义densenet模型 -*-*-
# -*- 定义开头前两层 -*-
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),       # padding是怎么判断取多少的？
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)       # 输出channel数是64，接下来将输入到第一个dense block
)

# -*- 逐渐添加各个dense block -*-
num_channels, growth_rate = 64, 32       # num_channels即第一个dense block的in_channels是64；growth_rate即卷积块的输出通道数为32
num_convs_in_dense_blocks = [4,4,4,4]           # 4个dense block，这里定义每个dense block里有4个卷积块

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    denseblock = Denseblock(num_convs=num_convs, in_channels=num_channels, out_channels=growth_rate)
    net.add_module(f'DenseBlock_{i}', denseblock)

    num_channels = denseblock.out_channels
    # 除了最后一个dense block之外，其他dense block都要在最后加一个过渡层
    if i != len(num_convs_in_dense_blocks)-1:
        transi_out_channels = num_channels // 2
        net.add_module(f'transition_block_{i}', transition_block(num_channels, transi_out_channels))
        num_channels = transi_out_channels

# -*- 最后接上全局池化层 -*-
net.add_module('BN', nn.BatchNorm2d(num_channels))
net.add_module('relu', nn.ReLU())
net.add_module('global avg pool', nn.AdaptiveAvgPool2d((1,1)))
# -*- 接上全连接层 -*-
net.add_module('fc', nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_channels, 10))
)


# -*- 查看网络 -*-
X = torch.rand((1,1,96,96))
for name, layer in net.named_children():
    X = layer(X)
    print(f'name:{name}, output shape:{X.shape}')




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

