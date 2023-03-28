import jieba
import warnings
import fasttext
from jieba import analyse
warnings.filterwarnings('ignore')
jieba.setLogLevel(jieba.logging.INFO)

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

with open("data/keywords_textrank.txt", "w", encoding='utf-8', errors='ignore') as f:
    for i in range(len(text)):
        keywords = jieba.analyse.textrank(texts[i], topK=10, allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'))
        f.writelines(str(texts[i]) + " " + " ".join(keywords) + '\n')
f.close()


