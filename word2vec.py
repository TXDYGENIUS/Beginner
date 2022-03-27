import re
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.optim import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import jieba
import random
from keras.layers import Input, Embedding, Lambda
from keras.models import Model
import keras.backend as K
from collections import Counter
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import word2vec
import logging
import torch.utils.data as tud
from collections import Counter
import scipy
from sklearn.metrics.pairwise import cosine_similarity

'''
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

C = 3  # context window
K = 15  # number of negative samples
epochs = 2
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 100
batch_size = 32
lr = 0.2

with open('normalTrafficTest.txt') as f:
    text = f.read()
text = text.lower()
text = re.split(' |,|/|=', text)
print(len(text))

vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))  # 得到单词字典表，key是单词，value是次数

vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))  # 把不常用的单词都编码为"<UNK>"
# values()为提取字典所有的值，list为把提取后的值转化为列表，sum为求列表中元素和
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(np.sum(list(vocab_dict.values())))
print(vocab_dict['<UNK>'])
print(len(text) - np.sum(list(vocab_dict.values())))
'''

with open('normalTrafficTest.txt') as f:
    text = f.read()
text = text.lower()
text = re.split(' |\n|/|=|:|,|;', text)

vocab_dict = dict(Counter(text).most_common(10000))  # 得到单词字典表，key是单词，value是次数

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Word2Vec第一个参数代表要训练的语料
# sg=1 表示使用Skip-Gram模型进行训练
# size 表示特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
# window 表示当前词与预测词在一个句子中的最大距离是多少
# min_count 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
# workers 表示训练的并行数
# sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)

def A():
    # 首先打开需要训练的文本
    shuju = open('F:/pythonProject/QQQ.txt', 'rb')
    # 通过Word2vec进行训练
    model = Word2Vec(LineSentence(shuju), sg=1, vector_size=100, window=10, min_count=5, workers=15, sample=1e-3)
    # 保存训练好的模型
    model.save('F:/pythonProject/word2vec.vector')

    print('训练完成')


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
import gensim

sentences = word2vec.Text8Corpus("QQQ.txt")

model = gensim.models.Word2Vec(sentences, sg=1, vector_size=100, window=5, min_count=2, negative=3, sample=0.001, hs=1,
                               workers=4)
model.wv.save_word2vec_format("QQ.model")

model = model.wv.load_word2vec_format('QQ.model')
a = model['text']  # len(model['text']) = 100


more_sentences = 'qwe'
model.train(more_sentences)