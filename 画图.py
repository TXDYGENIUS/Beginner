import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import csv
from torch import optim
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import torch.nn.parameter as parameter
import time
import math
import copy
from tqdm import tqdm

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

'''
def read():
    name_list = ['CNN', 'RNN', '决策树', '集成检测']
    # name_list = ['引言','相关工作','数据','方法','实验','结果','讨论']
    time1 = [91.20, 93.08, 90.08, 98.25]
    time2 = [90.53, 91.92, 88.62, 97.96]
    time3 = [94.67, 95.48, 94.70, 99.11]
    time4 = [92.55, 93.66, 91.59, 98.53]

    location = np.arange(len(name_list))
    width = 0.2
    plt.figure(figsize=(12, 8))
    plt.bar(location, time1, tick_label=name_list, width=width, label="Accuracy", alpha=0.8, color="w", edgecolor="k")
    for a, b in zip(location, time1):
        plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=10, weight='bold')
    plt.bar(location + width, time2, tick_label=name_list, width=width, label="Precision", alpha=0.8, color="w",
            edgecolor="k", hatch=".....")
    for a, b in zip(location + width, time2):
        plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    plt.bar(location + width * 2, time3, tick_label=name_list, width=width, label="Recall", alpha=0.8, color="w",
            edgecolor="k", hatch="/")
    for a, b in zip(location + width * 2, time3):
        plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    plt.bar(location + width * 3, time4, tick_label=name_list, width=width, label="F-Score", alpha=0.8, color="w",
            edgecolor="k", hatch="\\\\\\\\\\")
    for a, b in zip(location + width * 3, time4):
        plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=10)
    plt.ylim(80, 100)
    plt.legend(loc=2)

    def to_percent(temp, position):
        return '%1.0f' % (temp) + '%'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.show()


'''

# 设置默认字体，选择支持中文的字体以避免出现中文乱码情况
mpl.rcParams['font.sans-serif'] = ['SimHei']
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

lst_tempCNN = [99.01, 96.12, 95.21, 93.84, 94.99]
lst_tempGRU = [99.01, 96.95, 97.98, 97.15, 97.08]
lst_tempPNN = [99.01, 98.09, 97.56, 96.68, 98.83]
lst_temp = [96.92, 90.6, 91, 96.66]
input_values = ['Test_A', 'Test_B', 'Test_C', 'Test_D', 'Test_E']
fig, ax = plt.subplots()  # fig表示整张图片，ax表示图片中的各个图表
# ax.set_xlabel("分组", fontsize=14)  # 横坐标标签
# ax.set_ylabel("准确率", fontsize=14)  # 纵坐标标签
ax.plot(input_values, lst_tempCNN, marker='*', linestyle='--',label=u'Initial_model', color='k')  # 横坐标数据+纵坐标数据+图例
ax.plot(input_values, lst_tempGRU, marker='o', label=u'Adapt_model', color='k')
ax.plot(input_values, lst_tempPNN, marker='v', label=u'Train_model', color='k')
# ax.plot(input_values, lst_temp, marker='D', label=u'CGP', color='k')
plt.ylim(90, 100)  # 限定纵轴的范围


def to_percent(temp, position):
    return '%1.0f' % (temp) + '%'


plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.legend()  # 让图例生效
# 添加网格线
plt.grid(True, alpha=0.5, axis='both', linestyle=':')
plt.show()

all_number = 35000
