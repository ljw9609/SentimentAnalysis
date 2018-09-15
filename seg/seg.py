import os
import jieba
import pymysql
import re
from PIL import Image
import numpy as np
from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Seg(object):
    # stop_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stopwords.txt')
    stop_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/data/stopwords.txt'
    # db = pymysql.connect(host='localhost', user='root', password='123456', db='weiboSpider', port=3306, charset='utf8')
    db = pymysql.connect(host='101.132.180.255', user='root', password='uAiqwVwjJ8-i', db='viax', port=3306, charset='utf8')

    def __init__(self):
        self.seg_result = []

    def stopwordslist(self):
        stopwords = [line.strip() for line in open(self.stop_path, 'r', encoding='utf-8').readlines()]
        return stopwords

    def get_data_from_mysql(self, qty, offset):
        commentlist = []
        with self.db:
            cur = self.db.cursor()
            # cur.execute("select ctt from comments where id < '%d'" % qty)
            sql = "SELECT text FROM comments LIMIT %s OFFSET %s" % (qty, offset)
            cur.execute(sql)
            rows = cur.fetchall()
        reg = u"[\u4e00-\u9fa5]+"
        for row in rows:
            '''
            print(row[0])
            if row[0] == "":
                print("none")
                continue
            elif "@" in row[0]:
                print("@")
                continue
            '''
            res = re.findall(reg, row[0])
            if not res:
                continue
            # row = list(row)
            if res not in commentlist:
                commentlist.append(res[0])
        return commentlist

    def get_keyword_from_datalist(self, datalist):
        keyword = {}

        stopwords = self.stopwordslist()
        for sentence in datalist:
            seged_sentence = self.seg_sentence(sentence)
            words = list(set(seged_sentence) - set(stopwords))
            for word in words:
                keyword[word] = keyword.get(word, 0) + 1

        keyword = sorted(keyword.items(), key=lambda x: x[1], reverse=True)
        return keyword

    def seg_from_datalist(self, datalist):
        stopwords = self.stopwordslist()

        res = []

        for sentence in datalist:
            seged_sentence = self.seg_sentence(sentence)
            res.append(list(set(seged_sentence) - set(stopwords)))
        return res

    def seg_from_doc(self, doc):
        stopwords = self.stopwordslist()

        res = []
        s = doc.split('\n')

        for sentence in s:
            seged_sentence = self.seg_sentence(sentence)
            res.append(list(set(seged_sentence) - set(stopwords)))
        return res

    def seg_from_mysql(self, qty):
        datalist = self.get_data_from_mysql(qty)
        self.seg_result = self.seg_from_datalist(datalist)
        return self.seg_result

    @staticmethod
    def seg_sentence(sentence):
        sentence_seged = jieba.cut(sentence.strip())
        return sentence_seged



'''
def stopwordslist():
    stopwords = [line.strip() for line in open(stop_path, 'r', encoding='utf-8').readlines()]
    return stopwords


def seg_from_file(filename):
    stopwords = stopwordslist()

    f = open(filename, 'r', encoding='utf-8')

    line = f.readline()

    res = []

    while line:
        sentence = line.split('\n')[0]
        seged_sentence = seg_sentence(sentence)

        res.append(list(set(seged_sentence) - set(stopwords)))

        line = f.readline()
    print(res)
    return res


def seg_from_datalist(datalist):
    stopwords = stopwordslist()

    res = []

    for sentence in datalist:
        seged_sentence = seg_sentence(sentence)

        res.append(list(set(seged_sentence) - set(stopwords)))
    #print(res)
    return res

'''


def main():
    seg = Seg()
    doc = '''自然语言处理: 是人工智能和语言学领域的分支学科。
            在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
            自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。'''
    # res = seg.seg_from_doc(doc)
    datalist = seg.get_data_from_mysql(1000, 0)
    keywords = dict(seg.get_keyword_from_datalist(datalist))

    font_path = root_path + '/data/simfang.ttf'
    bg_path = root_path + '/data/bg.jpg'
    #back_color = np.array(Image.open(bg_path))
    back_color = imread(bg_path)
    image_colors = ImageColorGenerator(back_color)
    wordcloud = WordCloud(font_path=font_path, background_color="white", mask=back_color,
                          max_words=2000, max_font_size=100, random_state=48, width=1000, height=800, margin=2)
    wordcloud.generate_from_frequencies(keywords)
    plt.figure()
    plt.imshow(wordcloud.recolor(color_func=image_colors))
    plt.axis("off")
    plt.show()
    wordcloud.to_file(root_path + '/data/pic2.png')


if __name__ == '__main__':
    main()

