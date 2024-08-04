import random


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
