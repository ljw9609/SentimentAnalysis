import pickle
import gzip
import os
from math import exp, log
from collections import defaultdict
from seg.seg import Seg


class NaiveBayes(object):
    def __init__(self):
        self.corpus = {}
        self.counter = {}
        self.total = 0

    def train_model(self, data):
        print("------ Naive Bayes Classifier is training ------")
        for d in data:
            label = d[0]
            doc = d[1]
            if label not in self.corpus:
                self.corpus[label] = defaultdict(int)
                self.counter[label] = 0
            for word in doc:
                self.counter[label] += 1
                self.corpus[label][word] += 1
        self.total = sum(self.counter.values())
        print("------ Naive Bayes Classifier training over ------")

    def save_model(self, filename, iszip=True):
        print("------ Naive Bayes Classifier is saving model ------")
        d = {}
        d['counter'] = self.counter
        d['corpus'] = self.corpus
        d['total'] = self.total

        if not iszip:
            pickle.dump(d, open(filename, 'wb'), True)
        else:
            f = gzip.open(filename, 'wb')
            f.write(pickle.dumps(d))
            f.close()
        print("------ Naive Bayes Classifier saving model over ------")

    def load_model(self, filename, iszip=True):
        print("------ Naive Bayes Classifier is loading model ------")
        if not iszip:
            d = pickle.load(open(filename, 'rb'))
        else:
            try:
                f = gzip.open(filename, 'rb')
                d = pickle.loads(f.read())
            except IOError:
                f = open(filename, 'rb')
                d = pickle.loads(f.read())
            f.close()
        self.counter = d['counter']
        self.corpus = d['corpus']
        self.total = d['total']
        print("------ Naive Bayes Classifier model loading over ------")

    def predict(self, sentence):
        tmp = {}

        # 各特征中各类别的概率，如:P(negative|'讨厌')
        for k in self.corpus:

            # 各类别的概率，如:P(negative)
            tmp[k] = log(self.counter[k]) - log(self.total)

            # 各类别中各特征的条件概率，如:P('讨厌'|negative)
            for word in sentence:
                x = float(self.corpus[k].get(word, 1)) / self.counter[k]
                tmp[k] += log(x)

        res, prob = 0, 0
        for i in self.corpus:
            cur = 0
            try:
                for j in self.corpus:
                    cur += exp(tmp[j] - tmp[i])
                cur = 1.0 / cur
            except OverflowError:
                cur = 0.0
            if cur > prob:
                res, prob = i, cur
        return res, prob


def main():
    nb = NaiveBayes()
    data = [('neg', ['不', '喜欢', '关晓彤', '吐', '真', '讨厌']),
            ('pos', ['好', '支持', '开心']),
            ('pos', ['放', '干净', '点', '嘴巴', '个人观点']),
            ('pos', ['嘻嘻', '话', '这句', '赞同']),
            ('pos', ['加油']),
            ('neg', ['峰峰', '可怜', '一个', '惦记着', '丑女']),
            ('neg', ['喜欢', '吐', '关晓彤', '不', '真', '讨厌'])]
    # nb.train_model(data)
    # nb.load_model('model1', True)

    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    pos = open(root_path+'/data/pos.txt', 'r', encoding='utf-8').read()
    neg = open(root_path+'/data/neg.txt', 'r', encoding='utf-8').read()
    seg_pos = Seg().seg_from_doc(pos)
    seg_neg = Seg().seg_from_doc(neg)
    train_data = []
    for k in seg_pos:
        train_data.append(('pos', k))
    for j in seg_neg:
        train_data.append(('neg', j))

    nb.train_model(train_data)
    print(nb.total)
    print(nb.counter)
    print(nb.corpus)
    print(nb.predict(['好', '开心', '支持']))
    # nb.save_model('naivebayes_model1', True)


if __name__ == '__main__':
    main()