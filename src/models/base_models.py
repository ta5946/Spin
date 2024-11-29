import random
import nltk
import spacy
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.wsd import lesk
from difflib import SequenceMatcher
from gensim import downloader
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.models.distances import *


# Constant models
class Constant:
    def __init__(self, constant):
        self.constant = constant

    def predict(self, out1, out2):
        score = self.constant
        prediction = self.constant
        return score, prediction, ''


class Random:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def predict(self, out1, out2):
        score = random.random()
        prediction = int(score >= self.threshold)
        return score, prediction, ''


# Lexical models
class Lemmas:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        nltk.download('wordnet')
        self.model = WordNetLemmatizer()

    def get_set(self, sentence):
        words = sentence.split()
        return {self.model.lemmatize(word) for word in words}

    def predict(self, out1, out2):
        set1 = self.get_set(out1)
        set2 = self.get_set(out2)

        score = overlap_coefficient(set1, set2)
        prediction = int(score >= self.threshold)
        return score, prediction, ''


class Stems:
    def __init__(self, threshold=0.3):
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
        return score, prediction, ''


# String models
class Levenshtein:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def predict(self, out1, out2):
        score = 1 - levenshtein_distance(out1, out2)
        prediction = int(score >= self.threshold)
        return score, prediction, ''


class Sequence:
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        nltk.download('wordnet')
        self.model = SequenceMatcher()

    def predict(self, out1, out2):
        self.model.set_seqs(out1, out2)

        score = self.model.ratio()
        prediction = int(score >= self.threshold)
        return score, prediction, ''


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
        return score, prediction, ''


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
        return score, prediction, ''


# Ontology models
def get_synsets(sentence):
    words = word_tokenize(sentence)
    synsets = [lesk(words, word) for word in words if wordnet.synsets(word)]
    return synsets


def get_sentence_similarity(synsets1, synsets2, synset_similarity_function):
    sentence_similarity = 0
    for synset1 in synsets1:
        max_synset_similarity = 0
        for synset2 in synsets2:
            if synset1.pos() != synset2.pos():
                continue
            synset_similarity = synset_similarity_function(synset1, synset2)
            if not synset_similarity:
                continue
            if synset_similarity > max_synset_similarity:
                max_synset_similarity = synset_similarity
        sentence_similarity += max_synset_similarity
    return sentence_similarity / len(synsets1)


class Path:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        nltk.download('wordnet')
        nltk.download('punkt')

    def predict(self, out1, out2):
        synsets1 = get_synsets(out1)
        synsets2 = get_synsets(out2)
        if not synsets1 or not synsets2:
            score = 0
        else:
            sentence_similarity1 = get_sentence_similarity(synsets1, synsets2, path_similarity)
            sentence_similarity2 = get_sentence_similarity(synsets2, synsets1, path_similarity)
            score = (sentence_similarity1 + sentence_similarity2) / 2

        prediction = int(score >= self.threshold)
        return score, prediction, ''


class Lch:
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        nltk.download('wordnet')
        nltk.download('punkt')

    def predict(self, out1, out2):
        synsets1 = get_synsets(out1)
        synsets2 = get_synsets(out2)
        if not synsets1 or not synsets2:
            score = 0
        else:
            sentence_similarity1 = get_sentence_similarity(synsets1, synsets2, leacock_chodorow_similarity)
            sentence_similarity2 = get_sentence_similarity(synsets2, synsets1, leacock_chodorow_similarity)
            score = (sentence_similarity1 + sentence_similarity2) / 2

        prediction = int(score >= self.threshold)
        return score, prediction, ''


class Wup:
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        nltk.download('wordnet')
        nltk.download('punkt')

    def predict(self, out1, out2):
        synsets1 = get_synsets(out1)
        synsets2 = get_synsets(out2)
        if not synsets1 or not synsets2:
            score = 0
        else:
            sentence_similarity1 = get_sentence_similarity(synsets1, synsets2, wu_palmer_similarity)
            sentence_similarity2 = get_sentence_similarity(synsets2, synsets1, wu_palmer_similarity)
            score = (sentence_similarity1 + sentence_similarity2) / 2

        prediction = int(score >= self.threshold)
        return score, prediction, ''


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
        return score, prediction, ''


class BioBert:
    def __init__(self, threshold=0.9):
        self.tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
        self.model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2', device_map='cuda')
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
        return score, prediction, ''


class SentenceTransformers:
    def __init__(self, threshold=0.4):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device='cuda', trust_remote_code=True)
        self.threshold = threshold

    def get_vector(self, sentence):
        return self.model.encode(sentence)

    def predict(self, out1, out2):
        vector1 = self.get_vector(out1).reshape(1, -1)
        vector2 = self.get_vector(out2).reshape(1, -1)

        score = cosine_similarity(vector1, vector2)[0][0]
        prediction = int(score >= self.threshold)
        return score, prediction, ''
