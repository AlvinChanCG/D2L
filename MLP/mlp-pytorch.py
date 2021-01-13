import numpy as np
import sys
import torch
from torch import nn
from torch.nn import init
import torch
import torch.utils.data as Data
import torchvision
from torchvision import transforms as transforms

# train_data_root = "../softmax/DatasetsTemp/FashionMNIST/processed/training.pt"
# test_data_root = "../softmax/DatasetsTemp/FashionMNIST/processed/test.pt"
# train_mnist = torch.load(train_data_root)
# test_mnist = torch.load(test_data_root)

train_mnist = torchvision.datasets.FashionMNIST("../softmax/DatasetsTemp", train=True, download=False, transform=transforms.ToTensor())
test_mnist = torchvision.datasets.FashionMNIST("../softmax/DatasetsTemp", train=False, download=False, transform=transforms.ToTensor())

batchsize = 256
if sys.platform.startswith("win"):
    num_workers = 0
else:
    num_workers = 4
train_iter = Data.DataLoader(train_mnist, batch_size=batchsize, num_workers=num_workers, shuffle=True)
test_iter = Data.DataLoader(test_mnist, batch_size=batchsize, num_workers=num_workers, shuffle=False)

# 查看一下
# for x,y in train_iter:
#     print(x,y)
#     break

# -*-*- 定义模型 -*-*-
class Net(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(Net, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_hiddens, num_outputs)
        )

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y

num_inputs, num_hiddens, num_outputs = 784, 256, 10
net = Net(num_inputs, num_hiddens, num_outputs)

# -*-*- 定义损失函数 -*-*-
loss = nn.CrossEntropyLoss()

# -*-*- 定义优化器 -*-*-
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# -*-*- 训练模型 -*-*-
epochs = 5
for epoch in range(epochs):
    train_loss_sum, train_acc_num, test_acc_num, train_n, test_n = 0,0,0,0,0
    for x, y in train_iter:
        output = net(x)
        l = loss(output, y).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_loss_sum += l.item()
        train_acc_num += (output.argmax(dim=1)==y).sum().item()
        train_n += y.shape[0]
    # 测试集准确率
    for testx, testy in test_iter:
        test_output = net(testx)
        test_acc_num += (test_output.argmax(dim=1)==testy).sum().item()
        test_n += testy.shape[0]
    print(f"epoch:{epoch+1}, loss:{round(train_loss_sum/train_n,5)},"
          f"train acc: {round(train_acc_num/train_n, 3)},"
          f"test acc: {round(test_acc_num/test_n, 3)}")