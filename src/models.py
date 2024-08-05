import random
from nltk.stem import PorterStemmer
from utils import overlap_coefficient


class ConstantClassifier:
    def __init__(self, constant):
        self.constant = constant

    def predict(self, out1, out2):
        return self.constant


class RandomClassifier:
    def __init__(self):
        pass

    def predict(self, out1, out2):
        return random.choice([0, 1])


class StemsClassifier:
    def __init__(self, threshold):
        self.threshold = threshold
        self.model = PorterStemmer()

    def predict(self, out1, out2):
        stems1 = set([self.model.stem(word) for word in out1.split()])
        stems2 = set([self.model.stem(word) for word in out2.split()])
        set_similarity = overlap_coefficient(stems1, stems2)
        if set_similarity >= self.threshold:
            return 1
        else:
            return 0
