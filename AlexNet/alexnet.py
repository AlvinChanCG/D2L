import torch
import torchvision
import torch.utils.data as Data
from torch import nn
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -*-*- 定义模型（输入图像大小224x224x1） -*-*-
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,96,11,4),      # in_channel,out_channel,kernel_size,stride,padding; 输出96个54x54
            nn.ReLU(),
            nn.MaxPool2d(3,2),   # kernel_size, stride;输出96个
            nn.Conv2d(96,256,5,1,2),  # 输出256个
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            # 下面连续3个卷积层，且使用更小的kernel_size
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2)

        )

        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )

    def forward(self,x):
        feature = self.conv(x)
        y = self.fc(feature.view(x.shape[0],-1))   # 将卷积得到的FMs reshape成[batchsize, 256*5*5]
        return y


def load_data_fashion_mnist(batch_size, size=None, root='../softmax/DatasetsTemp'):
    trans = []   # 用于后面的transforms.Compose组合
    if size:
        trans.append(torchvision.transforms.Resize(size=size))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    train_data = torchvision.datasets.FashionMNIST(root=root,train=True,transform=transform,download=False)
    test_data = torchvision.datasets.FashionMNIST(root=root,train=False,transform=transform,download=False)
    import sys
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

def evaluate_acc(net, test_iter, ):
    net = net.to(device)
    test_n, test_correct_n = 0,0
    for x,y in test_iter:
        x = x.to(device)
        y = y.to(device)
        output = net(x)
        test_correct_n += (output.argmax(dim=1)==y).sum().cpu().item()
        test_n += y.shape[0]
    test_acc = test_correct_n/test_n
    return test_acc


if __name__ == '__main__':
    net = AlexNet()
    net = net.to(device)
    batchsize = 64
    epochs = 5
    lr = 0.001
    train_iter,test_iter = load_data_fashion_mnist(batch_size=batchsize,size=224)
    # -*-*- 损失函数 -*-*-
    loss = nn.CrossEntropyLoss()
    # -*-*- 优化器 -*-*-
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # -*-*- 训练 -*-*-
    print(f"training on {device}")
    for epoch in range(epochs):
        train_loss_sum, train_correct_sum, train_n = 0,0,0
        for x,y in train_iter:
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            l = loss(output,y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_correct_sum += (output.argmax(dim=1)==y).sum().cpu().item()
            train_n += y.shape[0]
        test_acc = evaluate_acc(net, test_iter)
        print(f"epoch:{epoch}, "
              f"trian avg loss:{round(train_loss_sum/train_n,4)}, "
              f"train acc:{round(train_correct_sum/train_n,3)}, "
              f"test acc:{round(test_acc, 3)}")




