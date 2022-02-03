import re
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.optim import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
import urllib.parse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

'''

normal = np.loadtxt("real_normal.txt")
anomalous = np.loadtxt("real_anomalous.txt")

print(normal)

standardScalar = StandardScaler()
X_train = standardScalar.fit_transform(normal)  # 标准化为每一竖列，标准后的值 = (标准前的值-平均值)/该竖列的标准差

print(X_train[0])
'''

learn = 0.01
size = 32
m = nn.ReLU(inplace=True)


def train(ano, data):
    z1 = torch.matmul(w1, data.T) + b1

    a1 = torch.tanh(z1)
    z2 = torch.matmul(w2, a1) + b2
    a2 = torch.tanh(z2)
    z3 = torch.matmul(w3, a2) + b3
    a3 = torch.sigmoid(z3)
    # z4 = torch.matmul(w4, a3) + b4
    # a4 = torch.sigmoid(z4)
    loss = (-(ano * torch.log(a3) + (1 - ano) * torch.log(1 - a3))).mean()
    if w1.grad is not None:
        w1.grad = None
    if w2.grad is not None:
        w2.grad = None
    if w3.grad is not None:
        w3.grad = None
    # if w4.grad is not None:
    #    w4.grad = None
    if b1.grad is not None:
        b1.grad = None
    if b2.grad is not None:
        b2.grad = None
    if b3.grad is not None:
        b3.grad = None
    # if b4.grad is not None:
    #    b4.grad = None
    loss.backward()
    w1.data = w1.data - w1.grad * learn
    w2.data = w2.data - w2.grad * learn
    w3.data = w3.data - w3.grad * learn
    # w4.data = w4.data - w4.grad * learn
    b1.data = b1.data - b1.grad * learn
    b2.data = b2.data - b2.grad * learn
    b3.data = b3.data - b3.grad * learn
    # b4.data = b4.data - b4.grad * learn
    # print(loss)
    # print(a3, '\n', ano, '+++++++++++++++++++++')


def test(ano, data):
    z1 = torch.matmul(w1, data.T) + b1
    a1 = torch.tanh(z1)
    z2 = torch.matmul(w2, a1) + b2
    a2 = torch.tanh(z2)
    z3 = torch.matmul(w3, a2) + b3
    a3 = torch.sigmoid(z3)
    # z4 = torch.matmul(w4, a3) + b4
    # a4 = torch.sigmoid(z4)
    print(a3, '\n', ano, '**********')
    W = 0
    for i in range(size):
        if a3[0][i].item() > 0.5 and ano[i] == 1:
            W = W + 1
        if a3[0][i].item() < 0.5 and ano[i] == 0:
            W = W + 1
    # print(W)
    return W


# +++++++++++++++++++++++++++


normal = np.loadtxt("real_normal.txt")
anomalous = np.loadtxt("real_anomalous.txt")
'''
# 数据归一化
standardScalar = StandardScaler()
standardScalar.fit(X_train)
X_train = standardScalar.transform(X_train)
'''
all_requests = np.concatenate([normal, anomalous])
X = all_requests
standardScalar = StandardScaler()
X = standardScalar.fit_transform(X)

X = torch.tensor(X)
y_normal = np.zeros(shape=(normal.shape[0]), dtype='int')
y_anomalous = np.ones(shape=(anomalous.shape[0]), dtype='int')
y = np.concatenate([y_normal, y_anomalous])
y = torch.tensor(y)
# 划分测试集和训练集
data_dataset = TensorDataset(X, y)

XX = []
YY = []
train_loader = DataLoader(dataset=data_dataset, batch_size=size, shuffle=True)

for idx, datat in enumerate(train_loader):
    inputs, labels = datat
    XX.append(inputs)
    YY.append(labels)

X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2, random_state=666)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# 数据:
w1 = torch.randn([10, 11], requires_grad=True, dtype=float)
w11 = torch.randn([10, 11], requires_grad=True, dtype=float)
w1.data = (w1.data - w11.data)
w2 = torch.randn([4, 10], requires_grad=True, dtype=float)
w22 = torch.randn([4, 10], requires_grad=True, dtype=float)
w2.data = (w2.data - w22.data)
w3 = torch.randn([1, 4], requires_grad=True, dtype=float)
w33 = torch.randn([1, 4], requires_grad=True, dtype=float)
w3.data = (w3.data - w33.data)
# w4 = torch.rand([1, 2], requires_grad=True, dtype=float)
# w44 = torch.rand([1, 2], requires_grad=True, dtype=float)
# w4.data = (w4.data - w44.data)
b1 = torch.randn([10, 1], requires_grad=True, dtype=float)
b2 = torch.randn([4, 1], requires_grad=True, dtype=float)
b3 = torch.randn([1, 1], requires_grad=True, dtype=float)
WW = 0

if __name__ == '__main__':
    start = time.time()
    print(len(X_test), '+++++++++++++++++++')
    for i in range(len(X_train)):
        train(y_train[i], X_train[i])

    for j in range(len(X_test)):
        W = test(y_test[j], X_test[j])
        WW = WW + W

    print('TURE = ', WW / (len(X_test) * size))
    end = time.time()
    print("运行时间:%.2f秒" % (end - start))
# 权重初始化，输入归一化
