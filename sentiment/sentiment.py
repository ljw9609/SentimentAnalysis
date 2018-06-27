from utils.naiveBayes import NaiveBayes
from utils.svm import SVM
from seg.seg import Seg
import os


class Sentiment(object):
    def __init__(self, method):
        self.nb = NaiveBayes(None)
        self.svm = SVM(50, None)
        self.seg = Seg()
        self.method = method
    '''
    def train_model(self, posdata, negdata):
        data = []
        for k in posdata:
            data.append(('pos', k))
        for k in negdata:
            data.append(('neg', k))
        self.nb.train_model(data)
    '''
    def train_model(self, data):
        if self.method == 1:
            self.nb.train_model(data)
        elif self.method == 2:
            self.svm.train_model(data)

    def save_model(self, filename):
        if self.method == 1:
            self.nb.save_model(filename)
        elif self.method == 2:
            self.svm.save_model(filename)

    def load_model(self, filename):
        if self.method == 1:
            self.nb.load_model(filename)
        elif self.method == 2:
            self.svm.load_model(filename)

    def predict_doc_nb(self, sentence):
        print("------ Naive Bayes Classifier is predicting ------")
        seged_sentence = self.seg.seg_from_doc(sentence)
        res, prob = self.nb.predict(seged_sentence[0])
        if res == 'pos':
            return prob
        print("------ Naive Bayes Classifier predicting over ------")
        return 1 - prob

    def predict_doc_svm(self, sentence):
        print("------ SVM Classifier predicting over ------")
        prob = self.svm.predict_sentence(sentence)
        print("------ SVM Classifier predicting over ------")
        return prob

    def predict_datalist_nb(self, datalist):
        print("------ Naive Bayes Classifier is predicting ------")
        seged_datalist = self.seg.seg_from_datalist(datalist)

        result = []
        for data in seged_datalist:
            res, prob = self.nb.predict(data)
            if res == 'pos':
                result.append(prob)
            else:
                result.append(1 - prob)
        print("------ Naive Bayes Classifier predicting over ------")
        return result

    def predict_datalist_svm(self, datalist):
        print("------ SVM Classifier is predicting ------")
        result = self.svm.predict_datalist(datalist)
        print("------ SVM Classifier predicting over ------")
        return result

    def predict_sentence_doc(self, sentence):
        if self.method == 1:
            return self.predict_doc_nb(sentence)
        elif self.method == 2:
            return self.predict_doc_svm(sentence)

    def predict_datalist(self, datalist):
        if self.method == 1:
            return self.predict_datalist_nb(datalist)
        elif self.method == 2:
            return self.predict_datalist_svm(datalist)


def main():
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    pos = open(root_path+'/data/pos.txt', 'r', encoding='utf-8').read()
    neg = open(root_path+'/data/neg.txt', 'r', encoding='utf-8').read()
    seg_pos = Seg().seg_from_doc(pos)
    seg_neg = Seg().seg_from_doc(neg)
    sentiment = Sentiment()
    sentiment.load_model(root_path+'/data/naivebayes_model1')
    # sentiment.train_model(seg_pos, seg_neg)
    doc = "我好开心啊！支持鹿晗！" \
          "加油加油加油"


if __name__ == '__main__':
    main()
