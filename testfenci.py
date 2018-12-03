import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
# import gensim.models.keyedvectors as word2vec
from gensim.models import word2vec
import numpy as np
import math
import os
# coding=utf-8
import csv
import jieba
import pandas as pd

path = "resource\\xunlian"
filenames = []
for dirpath, dirnames, filename in os.walk(path):
    # for dir in dirnames:
    #     fulldir = os.path.join(dirpath,dir)
    #     print(fulldir)

    for file in filename:
        fullpath = os.path.join(dirpath, file)
        filenames.append(fullpath)
        # print(fullpath)

stopwords_file = "resource/stop_words.txt"
# outputfile = "resource/output/"
# print(outputfile+filename[0][25:])

f = open(stopwords_file, "r", encoding='utf-8')
result = list()
for line in f.readlines():
    line = line.strip()
    if not len(line):
        continue

    result.append(line)
f.close()

# with open("resource/stop_words2.txt", "w", encoding='utf-8') as fw:
#     for sentence in result:
#         sentence.encode('utf-8')
#         data = sentence.strip()
#         if len(data) != 0:
#             fw.write(data)
#             fw.write("\n")
# print("end")

# 整理停用词 去空行和两边的空格

stop_f = open(stopwords_file, "r", encoding='utf-8')
stop_words = list()
for line in stop_f.readlines():
    line = line.strip()
    if not len(line):
        continue

    stop_words.append(line)
# stopwords_file.close()

for file in filenames:
    f = open(file, "r")
    result = list()
    try:
        for line in f.readlines():
            line = line.strip()
            if not len(line):
                continue
            outstr = ''
            seg_list = jieba.cut(line, cut_all=False)
            for word in seg_list:
                if word not in stop_words:
                    if word != '\t':
                        outstr += word
                        outstr += " "
            # seg_list = " ".join(seg_list)
            result.append(outstr.strip())
        f.close()

        # outputer = outputfile+file[25:]
        # print(outputer)
        with open(file, "w", encoding='utf-8') as fw:
            for sentence in result:
                sentence.encode('utf-8')
                data = sentence.strip()
                if len(data) != 0:
                    fw.write(data)
                    fw.write("\n")

                fw.write("\n")
    except:
        pass

print("end")