#coding:utf-8
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
# import gensim.models.keyedvectors as word2vec
from gensim.models import word2vec
import numpy as np
import math
import os
import csv
import jieba
import pandas as pd
import jieba.posseg as pseg
import codecs


path = "resource/xunlian"
# xlpath = "resource\\sougou_data_all"


# word2vec
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                yield line.split()

    # def __iter__(self):
    #     for dirpath, dirnames, filename in os.walk(self.dirname):
    #         # for dir in dirnames:
    #         #     fulldir = os.path.join(dirpath,dir)
    #         #     print(fulldir)
    #
    #         for file in filename:
    #             fullpath = os.path.join(dirpath, file)
    #             for line in open(fullpath, 'r').readlines():
    #                 print(line)
    #                 line.encode('utf-8')
    #                 yield line.split()
    #             # print(fullpath)

# stop_words = "resource/stop_words.txt"
# stopwords = codecs.open(stop_words,'r',encoding='utf-8').readlines()
# stopwords = [ w.strip() for w in stopwords ]
#
# stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']

def tokenization(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # print(line.strip())
            result.extend(line.strip().split())
    return result


# sentences = MySentences(path)
modelname = "model/test2word.model"
# model = word2vec.Word2Vec(sentences, min_count=1)
# model.save(modelname)
model = word2vec.Word2Vec.load(modelname)

datapath1 = 'resource/test/11.txt'
datapath2 = 'resource/test/49.txt'

cpSentence1 = tokenization(datapath1)
cpSentence2 = tokenization(datapath2)

print(model.n_similarity(cpSentence1, cpSentence2))




