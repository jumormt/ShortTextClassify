import os
import sys
import numpy as np
import matplotlib
import scipy
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from gensim import corpora, models, similarities
import codecs
import jieba.posseg as pseg
from sklearn.decomposition import LatentDirichletAllocation

# stop_words = "resource/stop_words.txt"
# stopwords = codecs.open(stop_words,'r',encoding='utf8').readlines()
# stopwords = [ w.strip() for w in stopwords ]
#
# stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']

# def tokenization(filename):
#     result = []
#     with open(filename, 'r') as f:
#         text = f.read()
#         words = pseg.cut(text)
#     for word, flag in words:
#         if flag not in stop_flag and word not in stopwords:
#             result.append(word)
#     return result
#
# def tokenization2(line):
#     result = []
#     words = pseg.cut(line)
#     for word, flag in words:
#         if flag not in stop_flag and word not in stopwords:
#             result.append(word)
#     return result

def tokenization(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # print(line.strip())
            result.extend(line.strip().split())
    return result

if __name__ == "__main__":

    # 存储读取语料 一行预料为一个文档
    corpus = []

    # for line in open(u'resource/test.txt', 'r', encoding="utf8").readlines():
    #     corpus.append(line.strip())
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
            fullpath = os.path.join(dirpath, file)
            filenames.append(fullpath)
            strr = ""
            for line in open(fullpath, encoding='utf-8').readlines():
                if line:
                    strr = strr+" "+line.strip()

            corpus.append(strr)
            # print(fullpath)

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    # print(vectorizer)
    #
    #
    X = vectorizer.fit_transform(corpus)
    analyze = vectorizer.build_analyzer()
    word = vectorizer.get_feature_names()
    weight = X.toarray()
    weight = np.asarray(weight)

    # newquepath = 'resource/test/10.txt'
    # strr = ''
    # newque = []
    # for line in open(newquepath, encoding='utf-8').readlines():
    #     if line:
    #         strr = strr + " " + line.strip()
    # newque.append(strr)
    # newx = vectorizer.transform(newque)
    # newwei = np.asarray(newx.toarray())
    #
    # newque2path = 'resource/test/11.txt'
    # strr2 = ''
    # newque2 = []
    # for line in open(newque2path, encoding='utf-8').readlines():
    #     if line:
    #         strr2 = strr2 + " " + line.strip()
    # newque2.append(strr2)
    # newx2 = vectorizer.transform(newque2)
    # newwei2 = np.asarray(newx2.toarray())




    # # 计算TFIDF
    #
    # 该类会统计每个词语的tf-idf权值
    # transformer = TfidfTransformer()
    #
    # # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    # tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    #
    # # 获取词袋模型中的所有词语
    # word = vectorizer.get_feature_names()
    #
    # # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    # weight = tfidf.toarray()
    # weight = np.asarray(weight)
    # # weight = np.asarray(weight).astype(np.int32)

    # testpath = 'resource/test'
    # testfile = 'resource/test/10.txt'
    # testque = []
    # y = []
    # for dirpath, dirnames, filename in os.walk(testpath):
    #
    #     for file in filename:
    #         numm = int(file[:-4])
    #         if ((numm >= 10) and (numm <= 59)):
    #             y.append(0)
    #         elif ((numm > 59) and (numm <= 560)):
    #             y.append(1)
    #         else:
    #             y.append(2)
    #         fullpath = os.path.join(dirpath, file)
    #         strr = ''
    #         for line in open(fullpath, encoding='utf-8').readlines():
    #             if line:
    #                 strr = strr + " " + line.strip()
    #         testque.append(strr)
    #         # print(fullpath)
    # testx = vectorizer.transform(testque)
    # testwei = np.asarray(testx.toarray())

    #
    # # 打印特征向量文本内容
    # print
    # 'Features length: ' + str(len(word))
    # for j in range(len(word)):
    #     print
    #     word[j]
    #
    #     # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    # for i in range(len(weight)):
    #     for j in range(len(word)):
    #         print
    #         weight[i][j],
    #     print
    #     '\n'

    n_topics = 3
    lda = LatentDirichletAllocation(n_topics=n_topics,
                                    max_iter=50,
                                    learning_method='batch')
    lda.fit(weight)  # tf即为Document_word Sparse Matrix

    doc_topic = lda.fit_transform(weight)
    # a0 = [0, 0, 0]
    # a1 = [0, 0, 0]
    # a2 = [0, 0, 0]
    # for i in range(len(doc_topic)):
    #     tm = doc_topic[i].argmax()
    #     if(yy[i] == 0):
    #         a0[tm] = a0[tm]+1
    #     elif(yy[i] == 1):
    #         a1[tm] = a1[tm]+1
    #     else:
    #         a2[tm] = a2[tm]+1
    # y0 = np.array(a0).argmax()
    # y1 = np.array(a1).argmax()
    # y2 = np.array(a2).argmax()
    # yn = [y0, y1, y2]

    topic_word = lda.components_
    # scoree = lda.score(newwei2)
    # scoree2 = lda.score(newwei)

    # newdoc_topic = lda.transform(newwei)
    # newdoc_topic2 = lda.transform(newwei2)
    # testdoc_topic = lda.transform(testwei)
    # testlabel = []
    # for i in range(len(testdoc_topic)):
    #     topic_most_pr = testdoc_topic[i].argmax()
    #     testlabel.append(topic_most_pr)
    #     # print("doc: {} topic: {}".format(n, topic_most_pr))
    #
    # trueecount = 0
    # for i in range(len(testdoc_topic)):
    #     if(yn[y[i]] == testlabel[i]):
    #         trueecount = trueecount+1
    # print(trueecount)


    n_top_words = 3
    # topic_wordc = topic_word.copy()
    #
    # topic_word = []
    #
    # for i in topic_wordc:
    #     si = []
    #     for j in i:
    #         si.append(j/sum(i))
    #     topic_word.append(si)
    #
    # topic_word = np.asarray(topic_word)
    # print(topic_word)

    # 打印每个主题下的核心词概率分布
    # 打印每个主题下权重较高的term
    for topic_idx, wordp in enumerate(topic_word):
        # wordpn = []
        # for i in wordp:
        #     i = i/sum(wordp)
        #     wordpn.append(i)
        print("Topic #%d:" % topic_idx)
        print(sum(wordp))
        # print(" ".join([word[i] for i in wordp.argsort()[:-n_top_words - 1:-1]]))
        # res = [(i,j) for i in word for j in wordp]
        print(list(zip(word,wordp)))
    print
        # 打印主题-词语分布矩阵
    # print
    # lda.components_




    # # 每个主题对应的问题词权重分布
    # import matplotlib.pyplot as plt
    #
    # f, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    # for i, k in enumerate([0, 1]):  # 两个主题
    #     ax[i].stem(topic_word[k, :], linefmt='b-',
    #                markerfmt='bo', basefmt='w-')
    #     ax[i].set_xlim(-2, 20)
    #     ax[i].set_ylim(0, 0.25)
    #     ax[i].set_ylabel("Prob")
    #     ax[i].set_title("topic {}".format(k))
    #
    # ax[1].set_xlabel("word")
    #
    # plt.tight_layout()
    # plt.show()

    # 输出前10篇文章最可能的Topic
    # label = []
    # for n in range(10):
    #     topic_most_pr = doc_topic[n].argmax()
    #     label.append(topic_most_pr)
    #     print("doc: {} topic: {}".format(n, topic_most_pr))
    #
    # import matplotlib.pyplot as plt
    #
    # f, ax = plt.subplots(6, 1, figsize=(8, 8), sharex=True)
    # for i, k in enumerate([0, 1, 2, 3, 8, 9]):
    #     ax[i].stem(doc_topic[k, :], linefmt='r-',
    #                    markerfmt='ro', basefmt='w-')
    #     ax[i].set_xlim(-1, 2)  # x坐标下标
    #     ax[i].set_ylim(0, 1.2)  # y坐标下标
    #     ax[i].set_ylabel("Prob")
    #     ax[i].set_title("Document {}".format(k))
    # ax[5].set_xlabel("Topic")
    # plt.tight_layout()
    # plt.show()

    import warnings

    warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
    # import gensim.models.keyedvectors as word2vec
    from gensim.models import word2vec
    import math


    class MySentences(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            for fname in os.listdir(self.dirname):
                strr = ''
                for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                    if line:
                        strr = strr+' '+line.strip()
                yield strr.split()

    # modelname = "model/test1.model"
    # sentences = MySentences(path)
    modelname = "model/ldaa.model"
    # model = word2vec.Word2Vec(sentences, min_count=1)
    # model.save(modelname)
    model = word2vec.Word2Vec.load(modelname)



    # def cos_sim(vector_a, vector_b):
    #     """
    #     计算两个向量之间的余弦相似度
    #     :param vector_a: 向量 a
    #     :param vector_b: 向量 b
    #     :return: sim
    #     """
    #     vector_a = np.mat(vector_a)
    #     vector_b = np.mat(vector_b)
    #     num = float(vector_a * vector_b.T)
    #     denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    #     cos = num / denom
    #     sim = 0.5 + 0.5 * cos
    #     return sim

    def dot_product(v1, v2):
        return sum(a * b for a, b in zip(v1, v2))


    def magnitude(vector):
        return math.sqrt(dot_product(vector, vector))


    def cos_sim2(v1, v2):
        '''计算余弦相似度
        '''
        return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + .00000000001)


    corpus2 = []
    # 测试
    testpath = 'resource/test'
    testfile = 'resource/test/10.txt'
    tfilenames = []
    y = []
    for dirpath, dirnames, filename in os.walk(testpath):

        for file in filename:
            numm = int(file[:-4])
            if ((numm >= 10) and (numm <= 59)):
                y.append(0)
            elif ((numm > 59) and (numm <= 560)):
                y.append(1)
            else:
                y.append(2)
            fullpath = os.path.join(dirpath, file)
            tfilenames.append(fullpath)
            corpus2.append(tokenization(fullpath))
            # print(fullpath)

    # co1 = corpus2[0]
    # co2 = corpus2[1]
    vecl = []
    for co in corpus2:
        cop = []
        for i in co:
            p = 0
            if i in word:
                # print(i)
                argi = word.index(i)
                for j in range(len(topic_word)):
                    p = p + (topic_word[j][argi] / sum(topic_word[j])) * doc_topic[0][j]
                # print(p)
            cop.append(p)

        veclist = []
        for i in range(len(cop)):
            if (co[i] in word):
                # print(co1[i])
                veclist.append(model[co[i]] * cop[i])
        vec = sum(veclist) / sum(cop)
        vecl.append(vec)

    truecount = 0
    count = 0
    for i in range(len(vecl)):
        for j in range(i+1, len(vecl)):
            count = count+1
            sim = cos_sim2(vecl[i], vecl[j])
            if(y[i] == y[j]):
                if(sim>0.5):
                    truecount = truecount+1
            else:
                if(sim<0.5):
                    truecount = truecount+1

    print(truecount)
    print(count)



    # co1p = []
    # for i in co1:
    #     p = 0
    #     if i in word:
    #         print(i)
    #         argi = word.index(i)
    #         for j in range(len(topic_word)):
    #             p = p + (topic_word[j][argi]/sum(topic_word[j]))*doc_topic[0][j]
    #         print(p)
    #     co1p.append(p)
    #
    # print(co1p)
    #
    # co2p = []
    # for i in co1:
    #     p = 0
    #     if i in word:
    #         argi = word.index(i)
    #         for j in range(len(topic_word)):
    #             p = p + (topic_word[j][argi]/sum(topic_word[j]))*doc_topic[1][j]
    #     co2p.append(p)
    #
    # vec1list = []
    # for i in range(len(co1p)):
    #     if (co1[i] in word):
    #         # print(co1[i])
    #         vec1list.append(model[co1[i]]*co1p[i])
    # vec1 = sum(vec1list)/sum(co1p)
    #
    # vec2list = []
    # for i in range(len(co2p)):
    #     if (co2[i] in word):
    #         # print(co2[i])
    #         vec2list.append(model[co2[i]] * co2p[i])
    # vec2 = sum(vec2list) / sum(co2p)
    #
    # print(cos_sim2(vec1, vec2))


    # print(cos_sim2(model['麦当劳'], model['曝光']))
    # print(model.similarity('麦当劳', '曝光'))
    #
    # print(cos_sim2(model['麦当劳'], model['制作']))
    # print(model.similarity('麦当劳', '制作'))
    #
    # print(cos_sim2(model['全过程'], model['制作']))
    # print(model.similarity('全过程', '制作'))
