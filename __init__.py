from seg import Seg
from sentiment import Sentiment
import os

root_path = os.path.abspath(os.path.dirname(__file__))


class SimpleNLP(object):
    def __init__(self, doc=None, datalist=None, stopword=False):
        self.doc = doc
        self.datalist = datalist
        self.stopword = []
        self.seg = Seg()
        self.sentiment = Sentiment()
        if stopword:
            self.stopword = self.seg.stopwordslist()

    def seg_datalist(self):
        return self.seg.seg_from_datalist(self.datalist)

    def seg_doc(self):
        return self.seg.seg_from_doc(self.doc)

    def sentiment_analysis_doc(self):
        self.sentiment.load_model(root_path+'/data/naivebayes_model1')
        return self.sentiment.predict_sentence(self.doc)


def main():
    doc = '''杰森我爱你！加油你是最棒的！'''
    datalist = Seg().get_data_from_mysql(100)
    npl = SimpleNLP(doc, True)
    res = npl.seg_doc()
    # res2 = npl.seg_datalist()
    print(res)
    # print(res2)
    sent = npl.sentiment_analysis_doc()
    print(sent)


if __name__ == '__main__':
    main()


