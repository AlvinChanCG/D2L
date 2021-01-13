import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from torch.nn import init

"""生成数据集"""
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
print('labels dtype:', labels.dtype)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),)

"""读取数据"""
batchsize = 10
dataset = Data.TensorDataset(features, labels)  # 将训练数据特征和标签组合, feature shape:[1000,2], label shape:[1000]
data_iter = Data.DataLoader(dataset, batchsize, shuffle=True)   # 注意DataLoader两个大写字母
# for x, y in data_iter:
#     print(x, y)



"""定义模型"""
class Network(nn.Module):
    def __init__(self, n_feature):
        super(Network, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_feature, 1)
        )
    # 前向传播
    def forward(self, x):
        #print(x.shape)
        y = self.linear(x)
        return y

net = Network(num_inputs)
print(net)
# 索引Sequential中的linear层
print(net.linear[0])


"""初始化模型参数（好像可以不用初始化）"""
print(net.linear[0].weight, '\n', net.linear[0].bias)
# init.normal_(net[0].weight, mean=0, std=0.01)
# init.constant_(net[0].bias, val=0)    # 或用net[0].bias.data.fill_(0)


"""定义损失函数"""
loss = nn.MSELoss()   # 默认输出的是标量形式的平均值(即reduction参数默认取mean)


"""定义优化算法"""
optimizer = optim.SGD(net.parameters(),lr=0.03)
print(optimizer)


"""训练"""
num_epochs = 3
for epoch in range(num_epochs):
    train_loss_sum = 0
    for index, (x, y) in enumerate(data_iter):
        if index == 0:
            print(f"x dtype:{x.dtype}, y dtype:{y.dtype}")   # 看一下数据类型
        x = x.clone().detach().float()              # 转成float类型，否则会报错
        # y = torch.tensor(y, dtype=torch.float32)    # 这样也可以
        y = y.clone().detach().float()
        output = net(x)   # 默认调用forward函数
        l = loss(output, y.view(-1, 1))    # 若y不view的话只有1维，view能转成两维，才能与output的两维对应上，否则虽不报错，但最后的loss不对
        train_loss_sum += l.item()*y.shape[0]
        optimizer.zero_grad()    # 梯度清零
        l.backward()
        optimizer.step()   # 迭代模型参数
    print(f"epoch {epoch}, loss:{l.item()}, train_average_loss:{train_loss_sum/features.shape[0]}")


"""查看参数回归结果"""
dense = net.linear[0]
print("\n")
print(true_w, '\n', dense.weight)
print(true_b, '\n', dense.bias)