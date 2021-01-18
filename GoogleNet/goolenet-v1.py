import torch
import torchvision
import torch.utils.data as Data
import time
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -*-*- 定义模型 -*-*-
# -*- 定义Inception块 -*-
class Inception(nn.Module):
    def __init__(self, in_channel, c1,c2,c3,c4):   # c1,c2,c3,c4分别为每条线路里的层的输出通道数
        super(Inception, self).__init__()
        # 线路1，只有1x1卷积层
        self.p1_1 = nn.Conv2d(in_channel, c1, kernel_size=1)
        # 线路2,1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channel, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3,1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channel, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4,3x3最大池化层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channel, c4, kernel_size=1)

    def forward(self,x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4), dim=1)      # 在通道维（）上连接输出


# -*- 定义全局平均池化（其实也可以不用） -*-
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self,x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# -*- 定义展平操作 -*-
class FlattenLayer(nn.Module):
    """
    function:对x进行reshape
    """
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self,x):
        #print(x.shape)
        return x.view(x.shape[0],-1)


# -*- 定义block(在主体卷积部分有5个block) -*-
block1 = nn.Sequential(
    nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

block2 = nn.Sequential(
    nn.Conv2d(64,64,kernel_size=1),
    nn.Conv2d(64,192,kernel_size=3,padding=1),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

block3 = nn.Sequential(
    Inception(in_channel=192, c1=64, c2=[96,128], c3=[16,32], c4=32),
    Inception(in_channel=256, c1=128, c2=[128,192], c3=[32,96], c4=64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

block4 = nn.Sequential(
    Inception(in_channel=480, c1=192, c2=[96,208], c3=[16, 48], c4=64),
    Inception(in_channel=512, c1=160, c2=[112,224], c3=[24,64], c4=64),
    Inception(in_channel=512, c1=128, c2=[128,256], c3=[24,64], c4=64),
    Inception(in_channel=512, c1=112, c2=[144,288], c3=[32,64], c4=64),
    Inception(in_channel=528, c1=256, c2=[160,320], c3=[32,128], c4=128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

block5 = nn.Sequential(
    Inception(in_channel=832, c1=256, c2=[160,320], c3=[32,128], c4=128),
    Inception(in_channel=832, c1=384, c2=[192, 384], c3=[48,128], c4=128),
    # nn.GlobalAvgPool2d(kernel_size=7, stride=2, padding=1)    # 自定义全局平均池化，也可以直接使用下面自适应平均池化
    nn.AdaptiveAvgPool2d((1,1))
)

# -*- 定义总体网络（方法1） -*-
net = nn.Sequential(
    block1,
    block2,
    block3,
    block4,
    block5,
    FlattenLayer(),
    nn.Linear(1024,10)
)


# -*- 定义总体网络的方法2 -*-
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block = nn.Sequential(
            block1,
            block2,
            block3,
            block4,
            block5
        )
        self.flatten = FlattenLayer()

    def forward(self,x):
        y = self.block(x)
        y = self.flatten(y)
        self.linear = nn.Linear(y.shape[1], 10)         # 不推荐在这里定义self.linear

        output = self.linear(y)
        return output
net3 = Net()


'''
# -*-*- 检查一下网络 -*-*-
X = torch.rand(1,1,96,96)
from copy import deepcopy
Y = deepcopy(X)
print(f"X is {X}, model output: {net3(X)}")
print(f"X is {X}, model output: {net(X)}")     # 为啥net3(x)和net(x)的输出不一样？？？

# -*- 打印每个模块的输出shape -*-
for block in net.children():
    X = block(X)
    print("net: output shape:", X.shape)
for blK in net3.children():
    Y = blK(Y)
    print("net3: output shape:", Y.shape)
'''

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
    for x,y in test_iter:
        output = net(x)
        test_correct_n = (output.argmax(dim=1)==y).sum()
        test_n += y.shape[0]
    test_acc = round(test_correct_n/test_n, 3)
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
    print()
    for epoch in range(epochs):
        train_loss_sum, train_n, train_correct_n = 0,0,0
        for x,y in train_iter:
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            l = loss(output, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_correct_n += (output.argmax(dim=1)==y).sum()
            train_n += y.shape[0]
        train_avg_loss = round(train_loss_sum/train_n, 4)
        train_acc = round(train_correct_n/train_n, 3)
        test_acc = evaluate_acc(test_iter, net)
        print(f"epoch:{epoch}, train avg loss:{train_avg_loss}, train acc:{train_acc}, test acc:{test_acc}")


