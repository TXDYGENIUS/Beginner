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
train_loader = DataLoader(dataset=data_dataset, batch_size=1, shuffle=True)

for idx, datat in enumerate(train_loader):
    inputs, labels = datat
    XX.append(inputs)
    YY.append(labels)

X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2, random_state=666)

XT = torch.tensor([item.detach().numpy() for item in X_train])  # torch.Size([48534, 1, 11])
YT = torch.tensor([item.detach().numpy() for item in y_train])  # torch.Size([48534, 1])
XX = torch.tensor([item.detach().numpy() for item in X_test])  # torch.Size([12134, 1, 11])
YY = torch.tensor([item.detach().numpy() for item in y_test])  # torch.Size([12134, 1])

XT = XT.reshape([(len(XT)), 11])
YT = YT.reshape([len(YT)])
XX = XX.reshape([len(XX), 11])
YY = YY.reshape([len(YY)])

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# xx, yy = make_classification(n_samples=(len(all_requests)), n_features=11, n_informative=2, n_redundant=0, n_classes=2,
#                            random_state=66,n_clusters_per_class=1)
# X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=666)
lr = LogisticRegression()
lr.fit(XT, YT)
y_hat = lr.predict(XX)
ww = torch.tensor(y_hat)
w = ww - YY
o = 0
for i in range(len(w)):
    if w[i] == 0:
        o = o + 1

print(o / len(w))
