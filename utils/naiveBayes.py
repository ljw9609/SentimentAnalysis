import pickle
import gzip
import os
from math import exp, log
from collections import defaultdict
from seg.seg import Seg
from utils.feature_extraction import FeatureExtraction
import time


class NaiveBayes(object):
    def __init__(self, best_words):
        self.corpus = {}
        self.counter = {}
        self.total = 0
        self.best_words = best_words

    def train_model(self, data):
        print("------ Naive Bayes Classifier is training ------")
        for d in data:
            label = d[0]
            doc = d[1]
            if label not in self.corpus:
                self.corpus[label] = defaultdict(int)
                self.counter[label] = 0
            for word in doc:
                if self.best_words is None or word in self.best_words:
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
        d['best_words'] = self.best_words

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
        if 'best_words' not in d:
            self.best_words = None
        else:
            self.best_words = d['best_words']
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
    start_time = time.time()
    nb = NaiveBayes(None)
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
    '''
    pos = open(root_path+'/data/pos.txt', 'r', encoding='utf-8').read()
    neg = open(root_path+'/data/neg.txt', 'r', encoding='utf-8').read()
    seg_pos = Seg().seg_from_doc(pos)
    seg_neg = Seg().seg_from_doc(neg)
    train_data = []
    for k in seg_pos:
        train_data.append(('pos', k))
    for j in seg_neg:
        train_data.append(('neg', j))
    '''
    nb.load_model(root_path+'/data/naivebayes_model20000v3')
    datalist = Seg().get_data_from_mysql(30000, 0)
    seged_datalist = Seg().seg_from_datalist(datalist)

    train_data = []
    doc_list = []
    doc_labels = []

    for data in seged_datalist:
        res, prob = nb.predict(data)
        if res == 'pos':
            doc_list.append(data)
            doc_labels.append('pos')
            train_data.append(('pos', data))
        else:
            doc_list.append(data)
            doc_labels.append('neg')
            train_data.append(('neg', data))
    print(train_data)
    fe = FeatureExtraction(doc_list, doc_labels)
    best_words = fe.best_words(5000, False)

    new_nb = NaiveBayes(best_words)
    new_nb.train_model(train_data)
    print(new_nb.predict(['好', '开心', '支持']))
    new_nb.save_model(root_path+'/data/naivebayes_model30000v3', True)
    '''
    nb.train_model(train_data)
    print(nb.total)
    print(nb.counter)
    print(nb.corpus)
    print(nb.predict(['好', '开心', '支持']))
    # nb.save_model('naivebayes_model1', True)
    '''
    end_time = time.time()
    print(end_time - start_time)


if __name__ == '__main__':
    main()