import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import jieba
import re
import os


# 输出主题词的文件路径
top_words_csv_path = 'data/top-topic-words.csv'
# 输出各文档所属主题的文件路径
predict_topic_csv_path = 'data/document-distribution.csv'

# 选定的主题数
n_topics = 14
# 要输出的每个主题的前 n_top_words 个主题词数
n_top_words = 20


def top_words_data(model: LatentDirichletAllocation,
                         tf_idf_vectorizer: TfidfVectorizer,
                         n_top_words: int) -> pd.DataFrame:
    '''
    求出每个主题的前 n_top_words 个词
    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation
    tf_idf_vectorizer : sklearn 的 TfidfVectorizer
    n_top_words :前 n_top_words 个主题词
    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    rows = []
    feature_names = tf_idf_vectorizer.get_feature_names()
    for topic in model.components_:
        top_words = [feature_names[i]
                     for i in topic.argsort()[:-500:-1]]
        rows.append(top_words)

    return rows


def predict_to_data(model: LatentDirichletAllocation, X: np.ndarray) -> pd.DataFrame:
    '''
    求出文档主题概率分布情况
    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation
    X : 词向量矩阵
    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    matrix = model.transform(X)
    return matrix.tolist()


texts = []
texts_14 = []
for i in range(14):
    texts_14.append([])
    with open("clusters_500/"+ str(i) +".txt", 'rb') as f:
        text = f.readlines()
        for j in range(len(text)):
            text[j] = text[j].decode('utf-8')
            texts.append(text[j])
            texts_14[i].append(text[j])
    f.close()

# 构造 tf-idf
tf_idf_vectorizer = TfidfVectorizer()
tf_idf = tf_idf_vectorizer.fit_transform(texts)

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=50,
    learning_method='online',
    learning_offset=50,
    random_state=0)

# 使用 tf_idf 语料训练 lda 模型
lda.fit(tf_idf)

# 计算 n_top_words 个主题词
top_words = top_words_data(lda, tf_idf_vectorizer, n_top_words)


# 转 tf_idf 为数组，以便后面使用它来对文本主题概率分布进行计算
X = tf_idf.toarray()

# 计算完毕主题概率分布情况
predict_list = predict_to_data(lda, X)

keywords = []
for i in range(len(texts)):
    topic_words = top_words[predict_list[i].index(max(predict_list[i]))]
    text_words = texts[i].split()
    words = [x for x in text_words if x in topic_words]
    s = list(set(words))
    c = []
    for j in s:
        c.append(words.count(j))
    s = [s[i] for i in np.array(c).argsort()[:-19:-1]]
    keywords.append(s)


with open("data/keywords.txt", 'w') as f:
    for j in keywords:
        f.writelines(" ".join(j) + '\n')
f.close()
