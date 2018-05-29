from utils.naiveBayes import NaiveBayes
from seg.seg import Seg
import os


class Sentiment(object):
    def __init__(self):
        self.nb = NaiveBayes()
        self.seg = Seg()

    def train_model(self, posdata, negdata):
        data = []
        for k in posdata:
            data.append(('pos', k))
        for k in negdata:
            data.append(('neg', k))
        self.nb.train_model(data)

    def save_model(self, filename):
        self.nb.save_model(filename)

    def load_model(self, filename):
        self.nb.load_model(filename)

    def predict_sentence(self, sentence):
        seged_sentence = self.seg.seg_from_doc(sentence)
        res, prob = self.nb.predict(seged_sentence[0])
        if res == 'pos':
            return prob
        return 1 - prob


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
    print(sentiment.predict_sentence(doc))


if __name__ == '__main__':
    main()
