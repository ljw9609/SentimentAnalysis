import os
import jieba
import pymysql


class Seg(object):
    stop_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stopwords.txt')
    db = pymysql.connect(host='localhost', user='root', password='123456', db='weiboSpider', port=3306, charset='utf8')

    def __init__(self):
        self.seg_result = []

    def stopwordslist(self):
        stopwords = [line.strip() for line in open(self.stop_path, 'r', encoding='utf-8').readlines()]
        return stopwords

    def get_data_from_mysql(self, qty):
        commentlist = []
        with self.db:
            cur = self.db.cursor()
            cur.execute("select text from comments3 where id < '%d'" % qty)
            rows = cur.fetchall()
        for row in rows:
            row = list(row)
            if row not in commentlist:
                commentlist.append(row[0])
        return commentlist

    def seg_from_datalist(self, datalist):
        stopwords = self.stopwordslist()

        res = []

        for sentence in datalist:
            seged_sentence = seg_sentence(sentence)
            res.append(list(set(seged_sentence) - set(stopwords)))
        return res

    def seg_from_doc(self, doc):
        stopwords = self.stopwordslist()

        res = []
        s = doc.split('\n')

        for sentence in s:
            seged_sentence = seg_sentence(sentence)
            res.append(list(set(seged_sentence) - set(stopwords)))
        return res

    def seg_from_mysql(self, qty):
        datalist = self.get_data_from_mysql(qty)
        self.seg_result = self.seg_from_datalist(datalist)
        return self.seg_result


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
    res = seg.seg_from_doc(doc)
    #res = seg.seg_from_mysql(20)
    print(res)


if __name__ == '__main__':
    main()

