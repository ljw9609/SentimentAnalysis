from sklearn.svm import SVC
from sklearn.externals import joblib
from utils.feature_extraction import FeatureExtraction
from seg.seg import Seg
from utils.naiveBayes import NaiveBayes
import numpy as np
import os
import pickle
import gzip


class SVM(object):
    def __init__(self, c, best_words):
        self.seg = Seg()
        self.clf = SVC(probability=True, C=c)
        self.train_data = []
        self.train_label = []
        self.best_words = best_words

    def words2vector(self, all_data):
        vectors = []
        for data in all_data:
            vector = []
            for feature in self.best_words:
                vector.append(data.count(feature))
            vectors.append(vector)
        vectors = np.array(vectors)
        return vectors

    def train_model(self, data):
        print("------ SVM Classifier is training ------")
        for d in data:
            label = d[0]
            doc = d[1]
            self.train_data.append(doc)
            self.train_label.append(label)

        self.train_data = np.array(self.train_data)
        self.train_label = np.array(self.train_label)

        train_vectors = self.words2vector(self.train_data)

        self.clf.fit(train_vectors, self.train_label)

        print("------ SVM Classifier training over ------")

    def save_model(self, filename):
        print("------ SVM Classifier is saving model ------")
        joblib.dump(self.clf, filename+'-model.m')
        f = gzip.open(filename + '-bestwords.dat', 'wb')
        d = {}
        d['best words'] = self.best_words
        f.write(pickle.dumps(d))
        f.close()
        print("------ SVM Classifier saving model over ------")

    def load_model(self, filename):
        print("------ SVM Classifier is loading model ------")
        self.clf = joblib.load(filename+'-model.m')

        f = gzip.open(filename+'-bestwords.dat', 'rb')
        d = pickle.loads(f.read())
        f.close()
        self.best_words = d['best words']
        print("------ SVM Classifier loading model over ------")

    def predict_wordlist(self, sentence):
        vector = self.words2vector([sentence])
        prediction = self.clf.predict(vector)
        prob = self.clf.predict_proba(vector)[0][1]
        return prediction[0], prob

    def predict_sentence(self, sentence):
        seged_sentence = self.seg.seg_from_doc(sentence)
        prediction, prob = self.predict_wordlist(seged_sentence)
        return prediction, prob

    def predict_datalist(self, datalist):
        seged_datalist = self.seg.seg_from_datalist(datalist)
        result = []
        for data in seged_datalist:
            prediction, prob = self.predict_wordlist(data)
            result.append(prob)
        return result


def main():
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    '''
    doc_list1 = [['不', '喜欢', '关晓彤', '吐', '真', '讨厌'], ['好', '支持', '开心'], ['放', '干净', '点', '嘴巴', '个人观点'],
                ['嘻嘻', '话', '这句', '赞同'], ['加油'], ['峰峰', '可怜', '一个', '惦记着', '丑女'], ['喜欢', '吐', '关晓彤', '不', '真', '讨厌']]
    doc_labels1 = ['neg', 'pos', 'pos', 'pos', 'pos', 'neg', 'neg']

    data = [('neg', ['不', '喜欢', '关晓彤', '吐', '真', '讨厌']),
            ('pos', ['好', '支持', '开心']),
            ('pos', ['放', '干净', '点', '嘴巴', '个人观点']),
            ('pos', ['嘻嘻', '话', '这句', '赞同']),
            ('pos', ['加油']),
            ('neg', ['峰峰', '可怜', '一个', '惦记着', '丑女']),
            ('neg', ['喜欢', '吐', '关晓彤', '不', '真', '讨厌'])]

    pos = open(root_path + '/data/pos.txt', 'r', encoding='utf-8').read()
    neg = open(root_path + '/data/neg.txt', 'r', encoding='utf-8').read()
    seg_pos = Seg().seg_from_doc(pos)
    seg_neg = Seg().seg_from_doc(neg)
    doc_list = []
    doc_labels = []
    train_data = []
    for k in seg_pos:
        train_data.append(('pos', k))
        doc_list.append(k)
        doc_labels.append('pos')
    for j in seg_neg:
        train_data.append(('neg', j))
        doc_list.append(j)
        doc_labels.append('neg')
    '''
    datalist = Seg().get_data_from_mysql(10000,0)
    print(datalist)
    nb = NaiveBayes(None)
    nb.load_model(root_path+'/data/naivebayes_model30000v3')
    seged_datalist = Seg().seg_from_datalist(datalist)

    train_data = []
    doc_list = []
    doc_labels = []
    word_list = []

    for data in seged_datalist:
        for word in data:
            word_list.append(word)
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

    print(best_words)
    print(word_list)
    svm = SVM(50, best_words)
    svm.train_model(train_data)

    svm.save_model(root_path + '/data/svmmodel10000v4')
    #svm.load_model(root_path + '/data/svmmodel4')


    #result = svm.predict_wordlist(['好', '喜欢', '加油', '赞同'])
    #result = svm.predict_sentence("加油")
    result = svm.predict_datalist(datalist)
    #print(result.count('pos'))
    print(result)


    #print(svm.train_label)
    #print(svm.train_data)


if __name__ == "__main__":
    main()