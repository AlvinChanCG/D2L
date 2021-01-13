import torch
from torch import nn
import torchvision
import torch.utils.data as Data
import pandas as pd
import numpy as np

torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv("./DATA/train.csv")
test_data = pd.read_csv("./DATA/test.csv")

# -*-*-*- 查看数据集 -*-*-*-
print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4, [0,1,2,3,-3,-2,-1]])   # 查看前4个样本的前4个、后2个特征，及标签（价格）
print(test_data.iloc[0:4, [0,1,2,3,-3,-2,-1]])

# -*-*-*- 拼接训练与测试数据，便于对所有数据进行处理 -*-*-*-
all_features = pd.concat((train_data.iloc[:,1:-1], test_data.iloc[:,1:]))   # 除去id字段，剩下的特征，包括训练集和测试集的
print(all_features.iloc[0:4, [0,1,2,3,-3,-2,-1]])

# -*-*-*- 预处理 -*-*-*-
# -*- 将连续数值(数值特征)进行标准化 -*-
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index   # 取字段是数值特征的列号索引
all_features[numeric_features] = all_features[numeric_features].apply(lambda x:(x-x.mean())/x.std())   # 将这些列进行标准化
# -*- 填充缺失值(用均值替换) -*-
print(all_features.iloc[0:5, 0:10])
all_features2 = all_features.fillna(0)    # 标准化后各个特征的均值就变成0
print(all_features2.iloc[0:5, 0:10])
# -*- 将离散数值转成指示特征(独热编码) -*-
all_features3 = pd.get_dummies(all_features2, dummy_na=True)
print(all_features3.iloc[0:5, 0:10])
print(all_features3.shape)
# -*- 通过values方法转成ndarray
num_train = train_data.shape[0]
ndarray_features_train = all_features3[:num_train].values
print(ndarray_features_train)
ndarray_features_test = all_features3[num_train:].values
# -*- 转tensor -*-
train_features = torch.tensor(ndarray_features_train[:num_train], dtype=torch.float)
test_features = torch.tensor(ndarray_features_test, dtype=torch.float)
# -*- 处理标签 -*-
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1,1)



# -*-*- 定义线性回归模型 -*-*-
class Net(nn.Module):
    def __init__(self,num_inputs, num_outputs):
        super(Net, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_inputs, num_outputs)
        )

    def forward(self,x):
        y = self.linear(x)
        return y

num_inputs, num_outputs = train_features.shape[1],1
net = Net(num_inputs, num_outputs).float()

# -*-*- 优化器 -*-*-
optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=0)

# -*-*- 损失函数 -*-*-
loss = nn.MSELoss(reduction='none')

# -*-*- 定义比赛规定的用来评价模型的误差 -*-*-
def log_rmse(net, features, labels):
    """
    :param net:
    :param features:
    :param labels:
    :return:
    """
    with torch.no_grad():
        # 将小于1的值设为1，使得取对数时数值更稳定  ???什么骚操作
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()).mean())
        return rmse.item()



# -*-*- k折交叉验证 -*-*-
# -*- 划分数据 -*-
def get_k_fold_data(k, i, x, y):
    """
    :param k:
    :param i: 表示取第i份作为验证集
    :param x:
    :param y:
    :return:
    """
    assert k>1
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j+1)*fold_size)
        x_part, y_part = x[idx], y[idx]
        if j==i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = torch.cat((x_train, x_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return x_train, y_train, x_valid, y_valid


def k_fold(k, x_train, y_train, num_epochs, batchsize):
    # -*-*- 训练 -*-*-
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        train_features, train_labels, x_valid, y_vaild = get_k_fold_data(k, i, x_train, y_train)

        # -*-*- 读取数据 -*-*-
        batch_size = batchsize
        dataset = Data.TensorDataset(train_features, train_labels)
        train_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_define_loss, valid_define_loss = [], []  # 记录每代的loss
        for epoch in range(num_epochs):
            for x, y in train_iter:
                l = loss(net(x.float()), y.float()).sum()     # 没有sum的话，原来的l是个二维tensor，而只有标量才能backward
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            train_define_loss.append(log_rmse(net=net, features=train_features, labels=train_labels))
            valid_define_loss.append(log_rmse(net=net, features=x_valid, labels=y_vaild))
        train_l_sum += train_define_loss[-1]   # 取最后一代的loss
        valid_l_sum += valid_define_loss[-1]
        print(f'fold:{i}, train rmse:{train_define_loss[-1]}, valid rmse:{valid_define_loss[-1]}')
    return train_l_sum/k, valid_l_sum/k, net

def pred(net, test_features, test_data):
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])       # [0]表示取第一行（虽然reshape之后只有一行）
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)


if __name__ == '__main__':
    k = 5
    num_epochs = 100
    batchsize = 64
    avg_train_loss, avg_val_loss, updated_net = k_fold(k, train_features, train_labels, num_epochs, batchsize)
    print(f"{k}-fold validation: avg train rmse:{avg_train_loss}, avg valid rmse:{avg_val_loss}")

    pred(net, test_features, test_data)

