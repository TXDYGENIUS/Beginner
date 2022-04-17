import re
import torch.nn as nn
import numpy as np
import torch

a = torch.rand(2, 3)  # 取值0-1
b = torch.randn(2, 3)  # 取值-1 - 1
print(a, '\n', b)
print(a.numpy())  # 转化成array
print(a.size(1))  # 获取第1个维度形状
a = a.view(1, 6)  # 改变形状
print(a.size())
a.dim()  # a 的阶数
a.max()  # a 中最大值
print(a.permute(1, 0))  # 交换第0个和第1个维度(高维度转置)
print(a[:, 1])  # 切片
print('++++++++++++++++++++++++++++')
a = torch.tensor(12)  # 表示数据本身
b = torch.Tensor(12)  # 传入数字表示形状
print(a, '\n', b)
c = torch.tensor(np.array(6))  # 指定数据类型
print(c)
a.add_(b)  # 若有’_‘的话则是直接修改a的值
