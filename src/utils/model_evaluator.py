import os
from time import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class ModelEvaluator:
    def __init__(self, model, data, out_file=None):
        self.model = model
        self.data = data
        self.n_rows = len(data)
        self.scores = []
        self.predictions = []
        self.evaluation_time = 0
        self.out_file = out_file

    def evaluate(self):
        start = time()

        for index, row in tqdm(self.data.iterrows(), total=self.n_rows, desc='Generating predictions'):
            out1 = row['out1']
            out2 = row['out2']
            score, prediction = self.model.predict(out1, out2)
            self.scores.append(score)
            self.predictions.append(prediction)

        end = time()
        self.evaluation_time = end - start

        if self.out_file:
            out_dir = os.path.dirname(self.out_file)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            self.data['score'] = self.scores
            self.data.to_csv(self.out_file, sep='\t', index=False)

    def get_metrics(self):
        labels = self.data['label']

        metrics = {
            'negative_ratio': self.predictions.count(0) / self.n_rows,
            'positive_ratio': self.predictions.count(1) / self.n_rows,
            'accuracy_score': accuracy_score(labels, self.predictions),
            'precision_score': precision_score(labels, self.predictions),
            'recall_score': recall_score(labels, self.predictions),
            'f1_score': f1_score(labels, self.predictions),
            'auc_score': roc_auc_score(labels, self.scores),
            'evaluation_time': self.evaluation_time
        }
        return metrics
