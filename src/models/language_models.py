import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


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


# TODO Large language models
