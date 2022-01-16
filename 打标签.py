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

size = 1
m = nn.ReLU(inplace=True)
learn = 0.01
A = np.loadtxt('vector_normal.txt')
B = np.loadtxt('vector_anomalous.txt')


#  A.shape  = (504000, 140)

# 把所有数据分成[14，140]的torch，在转换成[56.35]的矩阵存到列表里
def fenzu(A):
    a = []
    for i in range(int(len(A) / 14)):
        for j in range(14):
            w = torch.tensor(A[i * 14 + j])
            # w = w.reshape(1, 140)
            if j == 0:
                x = w
            else:
                x = torch.cat((x, w), 0)
        # x = x.reshape(56, 35)
        x = np.array(x)
        a.append(x)
    return a


normal = np.loadtxt("Normal")
anomalous = np.loadtxt("Anomalous")

all_requests = np.concatenate([normal, anomalous])  # 把两个矩阵第一个维度相加
X = all_requests
X = torch.tensor(X)
X = X.reshape(len(X), 56, 35)
# print(X.shape)=torch.Size([60668, 1960])
# print(X[0])
# print(X[0].shape)  =torch.Size([1960])


# print(XX[0].shape)=torch.Size([56, 35])
y_normal = np.zeros(shape=(normal.shape[0]), dtype='int')
y_anomalous = np.ones(shape=(anomalous.shape[0]), dtype='int')
y = np.concatenate([y_normal, y_anomalous])
y = torch.tensor(y)
# -----------------------
data_dataset = TensorDataset(X, y)

XX = []
YY = []
train_loader = DataLoader(dataset=data_dataset, batch_size=size, shuffle=True)
for idx, datat in enumerate(train_loader):
    inputs, labels = datat
    XX.append(inputs)
    YY.append(labels)

X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2, random_state=666)
'''
print(X_train[0])
print(X_train[0].shape, '+++++++++++++')  # = torch.Size([64, 56, 35])
print(y_train[0])
print(y_train[0].shape, '+++++++++++++')  # = torch.Size([64])
'''
w1 = torch.randn([20, 56], requires_grad=True, dtype=float)
w2 = torch.randn([10, 20], requires_grad=True, dtype=float)
w3 = torch.randn([4, 10], requires_grad=True, dtype=float)
w4 = torch.randn([1, 4], requires_grad=True, dtype=float)
b1 = torch.randn([20, 35], requires_grad=True, dtype=float)
b2 = torch.randn([10, 35], requires_grad=True, dtype=float)
b3 = torch.randn([4, 35], requires_grad=True, dtype=float)
b4 = torch.randn([1, 35], requires_grad=True, dtype=float)

ww1 = torch.randn([35, 10], requires_grad=True, dtype=float)
ww2 = torch.randn([10, 4], requires_grad=True, dtype=float)
ww3 = torch.randn([4, 1], requires_grad=True, dtype=float)
bb1 = torch.randn([1, 10], requires_grad=True, dtype=float)
bb2 = torch.randn([1, 4], requires_grad=True, dtype=float)
bb3 = torch.randn([1, 1], requires_grad=True, dtype=float)


def train(ano, data):
    z1 = torch.matmul(w1, data) + b1
    a1 = torch.tanh(z1)
    z2 = torch.matmul(w2, a1) + b2
    a2 = torch.tanh(z2)
    z3 = torch.matmul(w3, a2) + b3
    a3 = torch.tanh(z3)
    z4 = torch.matmul(w4, a3) + b4
    a4 = torch.tanh(z4)
    # -------------------------------
    zz1 = torch.matmul(a4, ww1) + bb1
    aa1 = torch.tanh(zz1)
    # print(zz1,'++++++++++++')
    zz2 = torch.matmul(aa1, ww2) + bb2
    aa2 = torch.tanh(zz2)
    # print(zz2,'----------')
    zz3 = torch.matmul(aa2, ww3) + bb3
    # print(zz3.shape)
    aa3 = torch.sigmoid(zz3)
    # aa3 = aa3.reshape(size,1)
    # print(aa3 ,'****************')
    loss = (-(ano * torch.log(aa3) + (1 - ano) * torch.log(1 - aa3))).mean()
    if w1.grad is not None:
        w1.grad = None
    if w2.grad is not None:
        w2.grad = None
    if w3.grad is not None:
        w3.grad = None
    if w4.grad is not None:
        w4.grad = None
    if b1.grad is not None:
        b1.grad = None
    if b2.grad is not None:
        b2.grad = None
    if b3.grad is not None:
        b3.grad = None
    if b4.grad is not None:
        b4.grad = None
    # -------------------------------
    if ww1.grad is not None:
        ww1.grad = None
    if ww2.grad is not None:
        ww2.grad = None
    if ww3.grad is not None:
        ww3.grad = None
    if bb1.grad is not None:
        bb1.grad = None
    if bb2.grad is not None:
        bb2.grad = None
    if bb3.grad is not None:
        bb3.grad = None

    loss.backward()
    w1.data = w1.data - w1.grad * learn
    w2.data = w2.data - w2.grad * learn
    w3.data = w3.data - w3.grad * learn
    w4.data = w4.data - w4.grad * learn
    b1.data = b1.data - b1.grad * learn
    b2.data = b2.data - b2.grad * learn
    b3.data = b3.data - b3.grad * learn
    b4.data = b4.data - b4.grad * learn
    ww1.data = ww1.data - ww1.grad * learn
    ww2.data = ww2.data - ww2.grad * learn
    ww3.data = ww3.data - ww3.grad * learn
    bb1.data = bb1.data - bb1.grad * learn
    bb2.data = bb2.data - bb2.grad * learn
    bb3.data = bb3.data - bb3.grad * learn
    print(loss)


def test(ano, data):
    W = 0
    z1 = torch.matmul(w1, data) + b1
    a1 = torch.tanh(z1)
    z2 = torch.matmul(w2, a1) + b2
    a2 = torch.tanh(z2)
    z3 = torch.matmul(w3, a2) + b3
    a3 = torch.tanh(z3)
    z4 = torch.matmul(w4, a3) + b4
    a4 = torch.tanh(z4)
    # -------------------------------
    zz1 = torch.matmul(a4, ww1) + bb1
    aa1 = torch.tanh(zz1)
    # print(zz1,'++++++++++++')
    zz2 = torch.matmul(aa1, ww2) + bb2
    aa2 = torch.tanh(zz2)
    # print(zz2,'----------')
    zz3 = torch.matmul(aa2, ww3) + bb3
    # print(zz3.shape)
    aa3 = torch.sigmoid(zz3)

    for i in range(size):
        if aa3[0][i].item() > 0.5 and ano[i] == 1:
            W = W + 1
        if aa3[0][i].item() < 0.5 and ano[i] == 0:
            W = W + 1
    # print(W)
    return W


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
    print("运行时间:%.2f秒"%(end-start))