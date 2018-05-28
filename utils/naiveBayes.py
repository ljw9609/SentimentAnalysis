import pickle
from math import exp, log
from collections import defaultdict


class NaiveBayes(object):
    def __init__(self):
        self.corpus = {}
        self.counter = {}
        self.total = 0

        