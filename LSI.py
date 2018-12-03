import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
import jieba.posseg as pseg
import codecs
from gensim import corpora, models, similarities
from gensim.models import word2vec

import os

# 遍历获取所有样本
path = 'resource/xunlian'
filenames = []
yy = []
for dirpath, dirnames, filename in os.walk(path):

    for file in filename:
        numm = int(file[:-4])
        if ((numm >= 10) and (numm <= 509)):
            yy.append(0)
        elif ((numm > 509) and (numm <= 1009)):
            yy.append(1)
        else:
            yy.append(2)
        fullpath = os.path.join(dirpath,file)
        filenames.append(fullpath)
        # print(fullpath)

def tokenization(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # print(line.strip())
            result.extend(line.strip().split())
    return result

# 建立词袋模型-begin
corpus = []
for each in filenames:
    # print(tokenization(each))
    corpus.append(tokenization(each))
# print (corpus[0])
dictionary = corpora.Dictionary(corpus)
# print(dictionary)

doc_vectors = [dictionary.doc2bow(text) for text in corpus]
# print (len(doc_vectors))
# print (doc_vectors)

# 建立词袋模型-end

# 建立TF-IDF模型-begin
tfidf = models.TfidfModel(doc_vectors)
tfidf_vectors = tfidf[doc_vectors]

# print (len(tfidf_vectors))
# print (len(tfidf_vectors[0]))


# 建立TF-IDF模型-end

# 测试
testpath = 'resource/test'
testfile = 'resource/test/10.txt'
tfilenames = []
y = []
for dirpath, dirnames, filename in os.walk(testpath):

    for file in filename:
        numm = int(file[:-4])
        if ((numm>=10) and (numm<=59)):
            y.append(0)
        elif ((numm>59)and (numm<=560)):
            y.append(1)
        else:
            y.append(2)
        fullpath = os.path.join(dirpath,file)
        tfilenames.append(fullpath)
        # print(fullpath)


query = tokenization(testfile)

query_bow = dictionary.doc2bow(query)

# 输出测试样本与训练样本的相似度
# index = similarities.MatrixSimilarity(tfidf_vectors)
# sims = index[query_bow]
# print (list(enumerate(sims)))

# 构建LSI模型，设置主题数为3
# modelname = 'model/lsimodel.model'
lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=3)
# lsi.save(modelname)
# lsi = models.LdaModel.load(modelname, mmap='r')
lsi_vector = lsi[tfidf_vectors]

# for vec in lsi_vector:
#     print(vec)# 维度相等


# 输出测试样本与训练样本的相似度
query_lsi = lsi[query_bow]
qlsi = []
for i in tfilenames:
    q = tokenization(i)
    qb = dictionary.doc2bow(q)
    qlsi.append(lsi[qb])

index = similarities.Similarity('Similarity-LSI-index', lsi_vector, num_features=400,num_best=3)
sims = index[qlsi]
truecount = 0
for i in range(len(sims)):
    if (yy[sims[i][0][0]] == y[i]) or (yy[sims[i][1][0]] == y[i]) or (yy[sims[i][2][0]] == y[i]):
        truecount = truecount+1

    # if y[i] == 0:
    #     if((sims[i][0][0]>=10 and sims[i][0][0]<=509) or (sims[i][1][0]>=10 and sims[i][1][0]<=509)) or (sims[i][2][0]>=10 and sims[i][2][0]<=509):
    #         truecount = truecount+1
    #         print(y[i], sims[i][0][0])
    #     else:
    #         pass
    # elif y[i] == 1:
    #     if (sims[i][0][0] >= 510 and sims[i][0][0] <= 1009) or (sims[i][1][0] >= 510 and sims[i][1][0] <= 1009) or (sims[i][2][0] >= 510 and sims[i][2][0] <= 1009):
    #         truecount = truecount + 1
    #         print(y[i], sims[i][0][0])
    #     else:
    #         pass
    # else:
    #     if (sims[i][0][0] >= 1010 and sims[i][0][0] <= 1509) or (sims[i][1][0] >= 1010 and sims[i][1][0] <= 1509) or (sims[i][2][0] >= 1010 and sims[i][2][0] <= 1509):
    #         truecount = truecount + 1
    #         print(y[i], sims[i][0][0])
    #     else:
    #         pass
print(truecount)
    # print(sims[i][0][0])
# print (list(enumerate(sims)))
