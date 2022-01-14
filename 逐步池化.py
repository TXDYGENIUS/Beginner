import re
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.optim import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
# 类似与卷积，每走一次池化一次
a = torch.rand(6, 6)
print(a)


def max(a, b, c, d):
    if a > b:
        b = a
    if c > d:
        d = c
    if d > b:
        b = d
    return b


def average(a, b, c, d, e, f, g, h, i):
    x = a + b + c + d + e + f + g + h + i
    return x / 9


def pool(a):
    lie = []
    for i in range(len(a) - 2):
        hang = []
        for j in range(len(a[0]) - 2):
            x = average(a[i][j], a[i][j + 1], a[i][j + 2], a[i + 1][j], a[i + 1][j + 1], a[i + 1][j + 2], a[i + 2][j],
                        a[i + 2][j + 1], a[i + 2][j + 2])
            hang.append(x)
        lie.append(hang)
    ss = torch.tensor(lie)
    print(ss)


pool(a)
