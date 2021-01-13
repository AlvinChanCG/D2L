import torch
import torchvision
from torchvision import transforms as transforms
from torch import nn
from torch.nn import init
import numpy as np
import torch.utils.data as Data
import sys
import os


data_root = "./DatasetsTemp"
if not os.path.exists(data_root):
    os.mkdir(data_root)
mnist_train = torchvision.datasets.FashionMNIST(root=data_root,train=True,download=True,transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root=data_root,train=False,download=True,transform=transforms.ToTensor())

# -*-*-*-*-*- 划分批量数据 *-*-*-*-*-*-*-
batch_size = 256
if sys.platform.startswith("win"):
    num_workers = 0
else:
    num_workers = 2
train_iter = Data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
test_iter = Data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)

# for x,y in train_iter:
#     print(x,y)
#     break

# -*-*-*-*-*- 初始化参数 -*-*-*-*-*-*-
num_inputs = 784
num_outputs = 10

# -*-*-*-*-*- 定义模型 -*-*-*-*-*-
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Net,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_inputs, num_outputs)
        )

    def forward(self,x):
        # 展平之后再输入
        y = self.linear(x.view(x.shape[0], -1))   # x.shape: [batchsize, c, h, w]
        return y

net = Net(num_inputs, num_outputs)

# -*-*-*-*-*- 定义损失函数 -*-*-*-*-*-
loss = nn.CrossEntropyLoss()   # 默认输出向量形式


# -*-*-*-*-*- 定义优化器 -*-*-*-*-*-
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# -*-*-*-*-*- 训练模型 -*-*-*-*-*-
num_epochs = 5
for epoch in range(num_epochs):
    train_l_sum, train_n, test_n = 0, 0, 0
    num_train_acc_sum, num_test_acc_sum = 0, 0
    for x,y in train_iter:
        output = net(x)      # output的shape:[batchsize, num_output], y的shape:[batchsize]
        l = loss(output, y).sum()   # 交叉熵默认输出向量形式
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        num_train_acc_sum += (output.argmax(dim=1)==y).sum().item()
        train_n += y.shape[0]
    # 每代之后，网络都会更新，测试一下结果
    for test_x, test_y in test_iter:
        num_test_acc_sum += (net(test_x).argmax(dim=1)==test_y).sum().item()
        test_n += test_y.shape[0]
    print(f'epoch:{epoch}, loss:{train_l_sum/train_n}, train acc:{num_train_acc_sum/train_n},'
          f'test acc:{num_test_acc_sum/test_n}')

