import random
import spacy
from nltk.stem import WordNetLemmatizer, PorterStemmer
from difflib import SequenceMatcher
from gensim import downloader
from sklearn.metrics.pairwise import cosine_similarity
from .distances import *


# Constant models
class Constant:
    def __init__(self, constant):
        self.constant = constant

    def predict(self, out1, out2):
        return self.constant


class Random:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        pass

    def predict(self, out1, out2):
        if random.random() >= self.threshold:
            return 1
        else:
            return 0


# Lexical models
class Lemmas:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = WordNetLemmatizer()

    def get_set(self, sentence):
        words = sentence.split()
        return {self.model.lemmatize(word) for word in words}

    def predict(self, out1, out2):
        set1 = self.get_set(out1)
        set2 = self.get_set(out2)

        set_similarity = overlap_similarity(set1, set2)
        if set_similarity >= self.threshold:
            return 1
        else:
            return 0


class Stems:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = PorterStemmer()

    def get_set(self, sentence):
        words = sentence.split()
        return {self.model.stem(word) for word in words}

    def predict(self, out1, out2):
        set1 = self.get_set(out1)
        set2 = self.get_set(out2)

        set_similarity = overlap_similarity(set1, set2)
        if set_similarity >= self.threshold:
            return 1
        else:
            return 0


# String models
class Levenshtein:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def predict(self, out1, out2):
        string_similarity = 1 - levenshtein_distance(out1, out2)
        if string_similarity >= self.threshold:
            return 1
        else:
            return 0


class Sequence:
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.model = SequenceMatcher()

    def predict(self, out1, out2):
        self.model.set_seqs(out1, out2)

        sequence_similarity = self.model.ratio()
        if sequence_similarity >= self.threshold:
            return 1
        else:
            return 0


# Vector models
class Spacy:
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


class WordVec:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = downloader.load('word2vec-google-news-300')

    def get_vector(self, sentence):
        words = sentence.split()
        return self.model.get_mean_vector(words)

    def predict(self, out1, out2):
        vector1 = self.get_vector(out1).reshape(1, -1)
        vector2 = self.get_vector(out2).reshape(1, -1)

        vector_similarity = cosine_similarity(vector1, vector2)[0][0]
        if vector_similarity >= self.threshold:
            return 1
        else:
            return 0


# TODO Ontology models
