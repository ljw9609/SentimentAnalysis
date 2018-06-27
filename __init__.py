from seg import Seg
from sentiment import Sentiment
import os
import numpy as np
from scipy.misc import imread
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import time
from collections import Counter
root_path = os.path.abspath(os.path.dirname(__file__))


class SimpleNLP(object):
    def __init__(self, method=1, doc=None, datalist=None):
        self.doc = doc
        self.datalist = datalist
        self.seg = Seg()
        self.sentiment = Sentiment(method)
        self.method = method

    def seg_datalist(self):
        return self.seg.seg_from_datalist(self.datalist)

    def seg_doc(self):
        return self.seg.seg_from_doc(self.doc)

    def get_keyword_datalist(self):
        return dict(self.seg.get_keyword_from_datalist(self.datalist))

    def sentiment_analysis_doc(self):
        if self.method == 1:
            self.sentiment.load_model(root_path+'/data/naivebayes_model30000v3')
        elif self.method == 2:
            self.sentiment.load_model(root_path+'/data/svmmodel10000v4')
        return self.sentiment.predict_sentence_doc(self.doc)

    def sentiment_analysis_datalist(self):
        if self.method == 1:
            self.sentiment.load_model(root_path+'/data/naivebayes_model30000v3')
        elif self.method == 2:
            self.sentiment.load_model(root_path+'/data/svmmodel10000v4')
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
    datalist = Seg().get_data_from_mysql(5, 0)
    npl = SimpleNLP(1, doc, datalist)
    print(npl.seg_doc())
    print(npl.seg_datalist())

    keyword = npl.get_keyword_datalist()
    print(keyword)
    print(len(keyword))
    '''
    font_path = root_path+'/data/simfang.ttf'
    bg_path = root_path + '/data/bg.jpg'
    back_color = imread(bg_path)
    image_colors = ImageColorGenerator(back_color)
    wordcloud = WordCloud(font_path=font_path, background_color="white", mask=back_color,
                          max_words=2000, max_font_size=100, width=1000, height=800, margin=2, random_state=48)
    wordcloud.generate_from_frequencies(keyword)
    plt.figure()
    plt.imshow(wordcloud.recolor(color_func=image_colors))
    plt.axis("off")
    plt.show()
    wordcloud.to_file(root_path + '/data/pic2.png')
    
    print(npl.sentiment_analysis_doc())
    res = npl.sentiment_analysis_datalist()
    # max_qty = Counter(res).most_common(1)[0][1]
    # print(max_qty)
    print(res)
    res2 = np.array(res)
    mean = np.mean(res2)
    print(mean)

    # plt.hist(res2, bins=np.arange(0, 1, 0.005))
    # plt.title("sentiment")
    # plt.xlabel("score")
    # plt.ylabel("amount")
    # plt.show()

    end_time = time.time()
    print(end_time - start_time)

    # f = open(root_path + '/data/10w-nb-30000v2', 'w')
    # f.write(str(res))
    # f.close()
    '''

if __name__ == '__main__':
    main()


