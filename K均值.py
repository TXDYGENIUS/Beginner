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
from numpy import *
from sklearn.neighbors import KNeighborsClassifier as kNN

'''

def distEclid(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))  # 求个数差的平方后和在开方


# 生成簇中心矩阵，每个簇中心向量值是样本每一维的平均值，初始情况下是随机值
def randCent(dataSet, k):
    # 获取数据集中的特征个数
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    # 遍历数据中的每一维
    for j in range(n):
        # 计算每一维的最大值和最小值，获得每一维的数据跨度，这样就可以生成数据范围内的随机数
        minJ = np.min(dataSet[:, j])
        rangeJ = float(np.max(dataSet[:, j]) - minJ)
        # 这里注意randint函数中的参数范围一定按大小顺序传入,此处使用random.rand(k,1)函数一定在文件头部写上如下语句
        # from numpy import *
        centroids[:, j] = np.mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


# k-均值聚类算法函数，后两个参数为计算两个向量之间的距离函数引用和计算簇中心的函数引用
def kMeans(dataSet, k, disMeas=distEclid, createCent=randCent):
    # 获得样本数量
    m = shape(dataSet)[0]
    # 定义矩阵，存储每个点的聚类标志和误差
    culsterAssment = mat(zeros((m, 2)))
    # 获取簇中心矩阵
    centroids = createCent(dataSet, k)
    # 簇中心变量是否变化标志
    clusterChanged = True
    # 迭代次数实现不确定，当簇中心矩阵不再变化时停止迭代
    while (clusterChanged):
        clusterChanged = False
        # 遍历每个样本
        for i in range(m):
            # 定义最小距离和最小距离所属的类索引
            minDist = np.inf
            minIndex = -1
            # 遍历所有的簇中心，计算该样本与所有簇中心的距离，比较获得距离最小的簇中心
            for j in range(k):
                distJI = disMeas(centroids[j, :], dataSet[i, :])
                # 更新最小距离和最近中心索引
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果样本聚类索引不等于计算出的最短距离中心索引，则继续迭代，并更新矩阵值
            if culsterAssment[i, 0] != minIndex:
                clusterChanged = True
            culsterAssment[i, :] = minIndex, minDist ** 2
        # print("centroids:", centroids)
        # 在迭代中，聚类变化，则簇中心变化，需要在每次迭代中更新簇中心
        for cent in range(k):
            # 过滤出已经聚类的样本
            ptsInClust = dataSet[nonzero(culsterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, culsterAssment
'''
size = 1

normal = np.loadtxt("real_normal.txt")
anomalous = np.loadtxt("real_anomalous.txt")
all_requests = np.concatenate([normal, anomalous])
X = all_requests  # len(all_requests) = 60668
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
# '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
trainmat = np.array([[1, 2, 3], [2, 3, 5], [55, 33, 66], [55, 33, 66]])
label = np.array([0, 0, 1, 1])

XT = torch.tensor([item.detach().numpy() for item in X_train])

YT = torch.tensor([item.detach().numpy() for item in y_train])
XX = torch.tensor([item.detach().numpy() for item in X_test])
YY = torch.tensor([item.detach().numpy() for item in y_test])

XT = (XT.reshape([48534, 11]))
YT = YT.reshape([48534])
XX = XX.reshape([12134, 11])
YY = YY.reshape([12134])

neigh = kNN(n_neighbors=3, algorithm='auto', weights='distance', n_jobs=1)
neigh.fit(XT, YT)

testmat = XX

answer = neigh.predict(testmat)
answer = torch.tensor(answer)
print(answer)
print('+++++++++++++++++++++++++++')
print(YY)

A = YY - answer
print(len(A))
aa = 0
for i in range(len(A)):
    if A[i] == 0:
        aa = aa + 1

print(aa/len(A))

'''
print(neigh.predict_proba(testmat))

print(X_train[0])
print(y_train[0])
    
'''
