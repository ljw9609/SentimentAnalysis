from seg.seg import *


class SimpleNLP(object):
    def __init__(self, doc=None, datalist=None, stopword=False):
        self.doc = doc
        self.datalist = datalist
        self.stopword = []
        if stopword:
            self.stopword = Seg().stopwordslist()

    def seg_datalist(self):
        return Seg().seg_from_datalist(self.datalist)

    def seg_doc(self):
        return Seg().seg_from_doc(self.doc)


def main():
    doc = '''自然语言处理:是人工智能和语言学领域的分支学科。
             在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
             自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。'''
    datalist = Seg().get_data_from_mysql(100)
    npl = SimpleNLP(doc, datalist, True)
    res = npl.seg_doc()
    res2 = npl.seg_datalist()
    print(res)
    print(res2)


if __name__ == '__main__':
    main()


