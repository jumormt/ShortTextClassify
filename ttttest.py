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

l = [1,2,3]

print(np.array(l).argmax())
