# SentimentAnalysis
Sentiment Analysis based on data crawled from [Sina Weibo](https://m.weibo.cn), using Natural Language Processing.

基于新浪微博数据的情感极性分析，使用机器学习算法训练模型，预计使用的方法包括朴素贝叶斯、K-NN、最大熵、SVM。

## 用法
````
doc = ```杰森我爱你！加油你是最棒的！```
nlp = SimpleNLP(doc, True)
# 分词
seg = nlp.seg_doc()
print('分词: ' + seg)
# 情感分析
sentiment = nlp.sentiment_analysis_doc()
print('情感分析: ' + sentiment)

# 结果
分词: [['棒', '杰森', '加油', '我爱你', '最']]
情感分析: 0.9364172150775273
````

## 实现功能
+ 分词（使用jieba分词库）
+ 情感分析（使用朴素贝叶斯算法训练模型）
+ 支持向量机（使用了sklearn工具）
+ 特征值提取（使用卡方检验算法）

## 详细说明
### (1)特征值提取
对应文件: feature_extraction.py

主要作用: 
+ 输入文本和对应的标签值
+ 使用卡方检验计算每个关键词的相关性
+ 根据需要输出有效关键词列表

核心算法: 卡方检验，排名越高代表特征相关度越高

举例: 考察特征词"喜欢"和类别"positive"的相关性

|特征选择|属于"positive"|不属于"positive"|总计|
|:---:|:---:|:---:|:---:|
|包含"喜欢"|A|B|A + B|
|不包含"喜欢"|C|D|C + D|
|总数|A + C|B + D|N|

则卡方("喜欢", "positive") = N(AD - BC)^2 / (A+C)(B+D)(A+D)(B+C)

### (2)支持向量机
对应文件: svm.py

主要作用:
+ 输入惩罚系数C、关键词列表和训练数据
+ 根据训练数据，构建词向量
+ 根据词向量和训练集标签，利用sklearn工具进行拟合
+ 输入测试词列表，预测分类

核心算法: 词向量构建、支持向量机

### (3)朴素贝叶斯
对应文件: naiveBayes.py

主要作用:
+ 输入训练数据集
+ 构建分类模型
+ 输入测试词列表，利用朴素贝叶斯算法计算各类别概率

核心算法: 朴素贝叶斯


# License
[MIT](https://github.com/ljw9609/SentimentAnalysis/blob/master/LICENSE)