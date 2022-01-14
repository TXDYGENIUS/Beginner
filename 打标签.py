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

m = nn.ReLU(inplace=True)

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
# print(X.shape)=torch.Size([60668, 1960])

y_normal = np.zeros(shape=(normal.shape[0]), dtype='int')
y_anomalous = np.ones(shape=(anomalous.shape[0]), dtype='int')
y = np.concatenate([y_normal, y_anomalous])
y = torch.tensor(y)
