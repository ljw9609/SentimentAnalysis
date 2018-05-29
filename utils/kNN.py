import pickle
import gzip
from collections import defaultdict


class KNN(object):
    def __init__(self, k=3):
        self.train_data_labels = []
        self.train_total_words = []
        self.train_total_words_length = 0
        self.k = k

    def train_model(self, data):
        for d in data:
            label = d[0]
            doc = d[1]
            self.train_data_labels.append(label)
            self.train_total_words.append(doc)
        self.train_total_words_length = len(self.train_total_words)

        vectors = []
