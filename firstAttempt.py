import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
import jieba.posseg as pseg
import codecs
from gensim import corpora, models, similarities
from gensim.models import word2vec

import os


path = r'resource\sougou_data_all'
filenames = []
for dirpath, dirnames, filenames in os.walk(path):
    # for dir in dirnames:
    #     fulldir = os.path.join(dirpath,dir)
    #     print(fulldir)

    for file in filenames:
        fullpath = os.path.join(dirpath,file)
        filenames.append(fullpath)
        # print(fullpath)


# filenames = ['resource/sougou_data_all/互联网/12.txt','resource/sougou_data_all/互联网/10.txt', 'resource/sougou_data_all/互联网/11.txt',
#              'resource/sougou_data_all/旅游/26.txt']

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

tfidf = models.TfidfModel(doc_vectors)
tfidf_vectors = tfidf[doc_vectors]

# print (len(tfidf_vectors))
# print (len(tfidf_vectors[0]))

testfile = 'resource/sougou_data_all/财经/999.txt'

query = tokenization(testfile)

query_bow = dictionary.doc2bow(query)

# index = similarities.MatrixSimilarity(tfidf_vectors)
#
# sims = index[query_bow]
# print (list(enumerate(sims)))

lsi = models.LsiModel(tfidf_vectors, id2word=dictionary, num_topics=9)
lsi_vector = lsi[tfidf_vectors]

for vec in lsi_vector:
    print(vec)

query_lsi = lsi[query_bow]

index = similarities.MatrixSimilarity(lsi_vector)
sims = index[query_lsi]
print (list(enumerate(sims)))

