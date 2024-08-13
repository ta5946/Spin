import random
import nltk
import spacy
import torch
from nltk.stem import WordNetLemmatizer, PorterStemmer
from difflib import SequenceMatcher
from gensim import downloader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from src.models.distances import *


# Constant models
class Constant:
    def __init__(self, constant):
        self.constant = constant

    def predict(self, out1, out2):
        score = self.constant
        prediction = self.constant
        return score, prediction


class Random:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def predict(self, out1, out2):
        score = random.random()
        prediction = int(score >= self.threshold)
        return score, prediction


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

        score = overlap_coefficient(set1, set2)
        prediction = int(score >= self.threshold)
        return score, prediction


class Stems:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        nltk.download('wordnet')
        self.model = PorterStemmer()

    def get_set(self, sentence):
        words = sentence.split()
        return {self.model.stem(word) for word in words}

    def predict(self, out1, out2):
        set1 = self.get_set(out1)
        set2 = self.get_set(out2)

        score = overlap_coefficient(set1, set2)
        prediction = int(score >= self.threshold)
        return score, prediction


# String models
class Levenshtein:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def predict(self, out1, out2):
        score = 1 - levenshtein_distance(out1, out2)
        prediction = int(score >= self.threshold)
        return score, prediction


class Sequence:
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        nltk.download('wordnet')
        self.model = SequenceMatcher()

    def predict(self, out1, out2):
        self.model.set_seqs(out1, out2)

        score = self.model.ratio()
        prediction = int(score >= self.threshold)
        return score, prediction


# Vector models
class Spacy:
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        spacy.cli.download('en_core_web_md')
        self.model = spacy.load('en_core_web_md')

    def predict(self, out1, out2):
        vector1 = self.model(out1).vector.reshape(1, -1)
        vector2 = self.model(out2).vector.reshape(1, -1)

        score = cosine_similarity(vector1, vector2)[0][0]
        prediction = int(score >= self.threshold)
        return score, prediction


class Word2Vec:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = downloader.load('word2vec-google-news-300')

    def get_vector(self, sentence):
        words = sentence.split()
        return self.model.get_mean_vector(words)

    def predict(self, out1, out2):
        vector1 = self.get_vector(out1).reshape(1, -1)
        vector2 = self.get_vector(out2).reshape(1, -1)

        score = cosine_similarity(vector1, vector2)[0][0]
        prediction = int(score >= self.threshold)
        return score, prediction


# TODO Ontology models


# Pretrained language models
class SciBert:
    def __init__(self, threshold=0.8):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', device_map='cuda')
        self.threshold = threshold

    def get_vector(self, sentence):
        inputs = self.tokenizer.encode(sentence)
        inputs = torch.tensor(inputs).reshape(1, -1).to('cuda')

        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs.last_hidden_state[:, 0, :].to('cpu')

    def predict(self, out1, out2):
        vector1 = self.get_vector(out1)
        vector2 = self.get_vector(out2)

        score = cosine_similarity(vector1, vector2)[0][0]
        prediction = int(score >= self.threshold)
        return score, prediction
