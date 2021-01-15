import torch
from torch import nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(    # 这里实现的时候与论文的网络模型图是不一样的
            nn.Conv2d(1,6,5),    # in_channel, out_channel, kernel_size 默认stride为1；输出6个24x24
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),    # kernel_size, stride；输出6个12x12
            nn.Conv2d(6,16,5),      # 输出16个8x8
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)    # 输出16个4x4的特征图
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),   #
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        feature = self.conv(x)
        y = self.fc(feature.view(feature.shape[0], -1))
        return y


def train(net, train_iter, test_iter, epochs, lr, device):
    net = net.to(device)
    print(f"training on {device}")
    # -*-*- 损失函数 -*-*-
    loss = nn.CrossEntropyLoss()
    # -*-*- 优化器 -*-*-
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss_sum, test_loss_sum, train_n, test_n = 0, 0, 0, 0
        train_correct_n = 0
        start = time.time()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            l = loss(output, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_n += y.shape[0]
            train_correct_n += (output.argmax(dim=1)==y).sum().cpu().item()
        test_acc = evaluate_acc(net,test_iter)
        print(f"epoch:{epoch}, train avg loss:{round(train_loss_sum/train_n,4)}, train acc:{round(train_correct_n/train_n,3)}, "
              f"test acc:{test_acc}, duration:{round(time.time()-start,3)} sec")


def evaluate_acc(net, test_iter, device='cuda' if torch.cuda.is_available() else 'cpu'):
    net = net.to(device)
    test_correct_n, test_n = 0, 0
    for test_x, test_y in test_iter:
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        assert isinstance(net, nn.Module)
        test_output = net(test_x)
        test_correct_n += (test_output.argmax(dim=1)==test_y).sum().cpu().item()
        test_n += test_y.shape[0]
    test_acc = round(test_correct_n/test_n, 3)
    return test_acc


if __name__ == '__main__':
    # -*-*- 实例化网络 -*-*-
    net = LeNet()
    print(net)

    # -*-*- 读取数据 -*-*-
    train_data = torchvision.datasets.FashionMNIST(root='../softmax/DatasetsTemp', train=True, download=False, transform=transforms.ToTensor())
    test_data = torchvision.datasets.FashionMNIST(root='../softmax/DatasetsTemp', train=True, download=False, transform=transforms.ToTensor())
    batch_size = 256
    import sys
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = Data.DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
    test_iter = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    # -*-*- 训练 -*-*-
    train(net=net, train_iter=train_iter, test_iter=test_iter, epochs=5,lr=0.001,device=device)