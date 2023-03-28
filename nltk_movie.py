import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


# 分析句子的情感：情感分析是NLP最受欢迎的应用之一。情感分析是指确定一段给定的文本是积极还是消极的过程。
# 有一些场景中，我们还会将“中性“作为第三个选项。情感分析常用于发现人们对于一个特定主题的看法。


# 定义一个用于提取特征的函数
# 输入一段文本返回形如：{'It': True, 'movie': True, 'amazing': True, 'is': True, 'an': True}
# 返回类型是一个dict
def extract_features(word_list):
    return dict([(word, True) for word in word_list])


# 我们需要训练数据，这里将用NLTK提供的电影评论数据
if __name__ == '__main__':
    # 加载积极与消极评论
    positive_fileids = movie_reviews.fileids('pos')  # list类型 1000条数据 每一条是一个txt文件
    negative_fileids = movie_reviews.fileids('neg')
    # print(type(positive_fileids), len(negative_fileids))

    # 将这些评论数据分成积极评论和消极评论
    # movie_reviews.words(fileids=[f])表示每一个txt文本里面的内容，结果是单词的列表：['films', 'adapted', 'from', 'comic', 'books', 'have', ...]
    # features_positive 结果为一个list
    # 结果形如：[({'shakesp: True, 'limit': True, 'mouth': True, ..., 'such': True, 'prophetic': True}, 'Positive'), ..., ({...}, 'Positive'), ...]
    features_positive = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in positive_fileids]
    features_negative = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in negative_fileids]

    # 分成训练数据集（80%）和测试数据集（20%）
    threshold_factor = 0.8
    threshold_positive = int(threshold_factor * len(features_positive))  # 800
    threshold_negative = int(threshold_factor * len(features_negative))  # 800
    # 提取特征 800个积极文本800个消极文本构成训练集  200+200构成测试文本
    features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
    print("\n训练数据点的数量:", len(features_train))
    print("测试数据点的数量:", len(features_test))

    # 训练朴素贝叶斯分类器
    classifier = NaiveBayesClassifier.train(features_train)
    print("\n分类器的准确性:", nltk.classify.util.accuracy(classifier, features_test))

    print("\n十大信息最丰富的单词:")
    for item in classifier.most_informative_features()[:10]:
        print(item[0])

    # 输入一些简单的评论
    input_reviews = [
        "It is an amazing movie",
        "This is a dull movie. I would never recommend it to anyone.",
        "The cinematography is pretty great in this movie",
        "The direction was terrible and the story was all over the place"
    ]
    # 运行分类器，获得预测结果
    print("\n预测:")
    for review in input_reviews:
        print("\n评论:", review)
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        # 打印输出
        print("预测情绪:", pred_sentiment)
        print("可能性:", round(probdist.prob(pred_sentiment), 2))


