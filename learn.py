import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
# import gensim.models.keyedvectors as word2vec
from gensim.models import word2vec
import numpy as np
import math

datapath = 'resource/data.txt'
modelname = "model/test1.model"

sentences = word2vec.Text8Corpus(datapath)
#
model = word2vec.Word2Vec(sentences,size=200)
#
# print(model.most_similar(positive=[u"奶粉"],topn=5))
# model.save(modelname)

# model = word2vec.Word2Vec.load(modelname)


# model.wv.save_word2vec_format("model/text8.model.bin",binary=True)

# model = word2vec.Word2Vec.load_word2vec_format("model/text8.model.bin", binary=True)
#
# print(model.wv.most_similar(positive=[u"奶粉"],topn=5))

# print(len(model[u'奶粉']))

# print(model.n_similarity(['麦当劳','曝光','麦乐鸡', '制作', '全过程'], ['吃', '小龙虾', '导致', '横纹肌', '溶解', '症']))

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



print (cos_sim2(model['麦当劳'],model['曝光']))
print(model.similarity('麦当劳','曝光'))

print (cos_sim2(model['麦当劳'],model['制作']))
print(model.similarity('麦当劳','制作'))

print (cos_sim2(model['全过程'],model['制作']))
print(model.similarity('全过程','制作'))