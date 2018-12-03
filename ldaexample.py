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

stop_words = "resource/stop_words.txt"
stopwords = codecs.open(stop_words,'r',encoding='utf8').readlines()
stopwords = [ w.strip() for w in stopwords ]

stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']

def tokenization(filename):
    result = []
    with open(filename, 'r') as f:
        text = f.read()
        words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result



def print_top_words(model, feature_names, n_top_words):
    # 打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print
        "Topic #%d:" % topic_idx
        print
        " ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]])
    print
        # 打印主题-词语分布矩阵
    print
    model.components_

if __name__ == "__main__":

    # 存储读取语料 一行预料为一个文档
    corpus = []

    # for line in open(u'resource/test.txt', 'r', encoding="utf8").readlines():
    #     words = pseg.cut(line)
    #     result = []
    #     for word,flag in words:
    #         if flag not in stop_flag and word not in stopwords:
    #             result.append(word)
    #     corpus.append(result)

    # print corpus

    # dictionary = corpora.Dictionary(corpus)
    # doc_vectors = [dictionary.doc2bow(text) for text in corpus]

    # tfidf = models.TfidfModel(doc_vectors)
    # corpus_tfidf = tfidf[corpus]
    # print(corpus_tfidf[0])

    for line in open(u'resource/test.txt', 'r', encoding="utf8").readlines():
        print(line)
        corpus.append(line.strip())



    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    print(vectorizer)


    X = vectorizer.fit_transform(corpus)
    # new = vectorizer.transform(单个doc)
    analyze = vectorizer.build_analyzer()
    weight = X.toarray()
    weight = np.asarray(weight)

    newque = ['大盘 春节 节目单']
    newx = vectorizer.transform(newque)
    newwei = np.asarray(newx.toarray())

    newque2 = ['新春 春节 年货 下跌']
    newx2 = vectorizer.transform(newque2)
    newwei2 = np.asarray(newx2.toarray())

    #
    # print (len(weight))
    # print(weight[:5, :5])


    # # 计算TFIDF
    #
    # # 该类会统计每个词语的tf-idf权值
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
    # weight = np.asarray(weight).astype(np.int32)

    n_topics = 2
    lda = LatentDirichletAllocation(n_topics=n_topics,
                                    max_iter=50,
                                    learning_method='batch')
    lda.fit(weight)  # tf即为Document_word Sparse Matrix

    doc_topic = lda.fit_transform(weight)
    topic_word = lda.components_
    # scoree = lda.score(newwei2)
    # scoree2 = lda.score(newwei)

    newdoc_topic = lda.transform(newwei)
    newdoc_topic2 = lda.transform(newwei2)


    # n_top_words = 15
    #
    # print_top_words(lda, word, n_top_words)

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

    #     # LDA算法
    # print
    # 'LDA:'
    # import numpy as np
    # import lda
    # import lda.datasets
    #
    # model = lda.LDA(n_topics=2, n_iter=500, random_state=1)
    # model.fit(weight)  # model.fit_transform(X) is also available
    # topic_word = model.topic_word_  # model.components_ also works
    #
    # # 文档-主题（Document-Topic）分布
    # doc_topic = model.doc_topic_
    # print("type(doc_topic): {}".format(type(doc_topic)))
    # print("shape: {}".format(doc_topic.shape))
    #
    # # 输出前10篇文章最可能的Topic
    # label = []
    # for n in range(10):
    #     topic_most_pr = doc_topic[n].argmax()
    #     label.append(topic_most_pr)
    #     print("doc: {} topic: {}".format(n, topic_most_pr))
    #
    # # 计算文档主题分布图
    # import matplotlib.pyplot as plt
    #
    # f, ax = plt.subplots(6, 1, figsize=(8, 8), sharex=True)
    # for i, k in enumerate([0, 1, 2, 3, 8, 9]):
    #     ax[i].stem(doc_topic[k, :], linefmt='r-',
    #                markerfmt='ro', basefmt='w-')
    #     ax[i].set_xlim(-1, 2)  # x坐标下标
    #     ax[i].set_ylim(0, 1.2)  # y坐标下标
    #     ax[i].set_ylabel("Prob")
    #     ax[i].set_title("Document {}".format(k))
    # ax[5].set_xlabel("Topic")
    # plt.tight_layout()
    # plt.show()

    # tfidf-每个主题对应的问题词权重分布
    # import matplotlib.pyplot as plt
    #
    # f, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    # for i, k in enumerate([0, 1]):  # 两个主题
    #     ax[i].stem(topic_word[k, :], linefmt='b-',
    #                markerfmt='bo', basefmt='w-')
    #     ax[i].set_xlim(-2, 20)
    #     ax[i].set_ylim(0, 5)
    #     ax[i].set_ylabel("Prob")
    #     ax[i].set_title("topic {}".format(k))
    #
    # ax[1].set_xlabel("word")
    #
    # plt.tight_layout()
    # plt.show()

    # # tfidf-输出前10篇文章最可能的Topic
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
