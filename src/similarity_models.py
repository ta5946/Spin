import random
from nltk.stem import WordNetLemmatizer, PorterStemmer
from difflib import SequenceMatcher
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from utils import overlap_coefficient, levenshtein_distance


# Base models
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


# Lexical measure models
class LemmasClassifier:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = WordNetLemmatizer()

    def predict(self, out1, out2):
        set1 = set([self.model.lemmatize(word) for word in out1.split()])
        set2 = set([self.model.lemmatize(word) for word in out2.split()])
        set_similarity = overlap_coefficient(set1, set2)
        if set_similarity >= self.threshold:
            return 1
        else:
            return 0


class StemsClassifier:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = PorterStemmer()

    def predict(self, out1, out2):
        set1 = set([self.model.stem(word) for word in out1.split()])
        set2 = set([self.model.stem(word) for word in out2.split()])
        set_similarity = overlap_coefficient(set1, set2)
        if set_similarity >= self.threshold:
            return 1
        else:
            return 0


# String measure models
class LevenshteinClassifier:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def predict(self, out1, out2):
        string_similarity = 1 - levenshtein_distance(out1, out2)
        if string_similarity >= self.threshold:
            return 1
        else:
            return 0


class SequenceClassifier:
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.model = SequenceMatcher()

    def predict(self, out1, out2):
        self.model.set_seqs(out1, out2)
        string_similarity = self.model.ratio()
        if string_similarity >= self.threshold:
            return 1
        else:
            return 0


# Vector measure models
class SpacyClassifier:
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.model = spacy.load('en_core_web_md')

    def predict(self, out1, out2):
        vector1 = self.model(out1).vector.reshape(1, -1)
        vector2 = self.model(out2).vector.reshape(1, -1)
        vector_similarity = cosine_similarity(vector1, vector2)[0][0]
        if vector_similarity >= self.threshold:
            return 1
        else:
            return 0
