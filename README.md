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

## 未完成工作
+ K-近邻分类算法（K-Nearest-Neighbours Algorithm）
+ 最大熵模型（Maximum Entropy）
+ 支持向量机（Support Vector Machines）

# License
[MIT](https://github.com/ljw9609/SentimentAnalysis/blob/master/LICENSE)