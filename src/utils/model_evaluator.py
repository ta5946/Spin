from time import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluator:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.n = len(data)
        self.predictions = []
        self.evaluation_time = 0

    def evaluate(self):
        start = time()

        for i, row in tqdm(self.data.iterrows(), total=self.n, desc='Generating predictions'):
            out1 = row['out1']
            out2 = row['out2']
            self.predictions.append(self.model.predict(out1, out2))

        end = time()
        self.evaluation_time = end - start

    def get_scores(self):
        labels = self.data['label']

        # TODO Auc score
        scores = {
            'negative_ratio': self.predictions.count(0) / self.n,
            'positive_ratio': self.predictions.count(1) / self.n,
            'accuracy:score': accuracy_score(labels, self.predictions),
            'precision_score': precision_score(labels, self.predictions),
            'recall_score': recall_score(labels, self.predictions),
            'f1_score': f1_score(labels, self.predictions),
            'evaluation_time': self.evaluation_time
        }
        return scores
