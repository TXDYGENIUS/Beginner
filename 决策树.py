import re
import jieba
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
import joblib
from keras.preprocessing.text import Tokenizer
from sklearn.datasets import make_classification
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import tree

A = np.loadtxt('vector_normal.txt')  # print(len(fenzu(A)[0])) = 1960
B = np.loadtxt('vector_anomalous.txt')


def fenzu(A):
    a = []
    for i in range(int(len(A) / 14)):
        for j in range(14):
            w = torch.tensor(A[i * 14 + j])
            if j == 0:
                x = w
            else:
                x = torch.cat((x, w), 0)
        x = np.array(x)
        a.append(x)
    return a


normal = np.array(fenzu(A))
anomalous = np.array(fenzu(B))

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
train_loader = DataLoader(dataset=data_dataset, batch_size=1, shuffle=True)

for idx, datat in enumerate(train_loader):
    inputs, labels = datat
    XX.append(inputs)
    YY.append(labels)

X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2, random_state=666)
# '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

XT = torch.tensor([item.detach().numpy() for item in X_train])
YT = torch.tensor([item.detach().numpy() for item in y_train])
XX = torch.tensor([item.detach().numpy() for item in X_test])
YY = torch.tensor([item.detach().numpy() for item in y_test])

XT = XT.reshape([(len(XT)), 1960])
YT = YT.reshape([len(YT)])
XX = XX.reshape([len(XX), 1960])
YY = YY.reshape([len(YY)])
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
model = tree.DecisionTreeClassifier(criterion='entropy', random_state=30, splitter='random')
model = model.fit(XT, YT)
y_hat = model.predict(XX)
y_hat = torch.tensor(y_hat)
score = model.score(XX, YY)
print(score)


ww = y_hat - YY
o = 0
for i in range(len(ww)):
    if ww[i] == 0:
        o = o + 1
print(o / len(ww))

