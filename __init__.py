from seg import Seg
from sentiment import Sentiment
import os
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import time
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

    def get_keyword_datalist(self):
        return self.seg.get_keyword_from_datalist(self.datalist)

    def sentiment_analysis_doc(self):
        self.sentiment.load_model(root_path+'/data/naivebayes_model1')
        return self.sentiment.predict_sentence_doc(self.doc)

    def sentiment_analysis_datalist(self):
        self.sentiment.load_model(root_path+'/data/naivebayes_model3')
        return self.sentiment.predict_datalist(self.datalist)


def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    plt.hist(myList,100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()


def main():
    doc = '''杰森我爱你！加油你是最棒的！'''
    start_time = time.time()
    datalist = Seg().get_data_from_mysql(50000)
    npl = SimpleNLP(None, datalist, True)
    keyword = dict(npl.get_keyword_datalist())
    print(keyword)
    print(len(keyword))

    '''
    stopwords = set(STOPWORDS)
    stopwords.add("不")
    stopwords.add("都")
    stopwords.add("还")
    font_path = root_path+'/data/simfang.ttf'
    wordcloud = WordCloud(font_path=font_path, background_color="white", stopwords=stopwords,
                          max_words=2000, max_font_size=100, width=1000, height=800)
    wordcloud.generate_from_frequencies(keyword)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    wordcloud.to_file(root_path+'/data/pic1.png')
    '''
    res = npl.sentiment_analysis_datalist()
    print(res)
    res = np.array(res)
    mean = np.mean(res)
    print(mean)

    draw_hist(res, "sentiment", "score", "amount", 0.0, 1.0, 0, 4000)

    end_time = time.time()
    print(end_time - start_time)

    #npl = SimpleNLP(doc, True)
    #res = npl.seg_doc()
    # res2 = npl.seg_datalist()
    #print(res)
    # print(res2)
    #sent = npl.sentiment_analysis_doc()
    #print(sent)


if __name__ == '__main__':
    main()


