import pandas as pd
import wandb
from dotenv import load_dotenv
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import similarity_models

load_dotenv()
DATA_FILE = '../data/outcome_similarity/train.tsv'
SIMILARITY_CLASSIFIER = similarity_models.SequenceClassifier(threshold=0.4)


wandb.login()
run = wandb.init(project='outcome_similarity_detection',
                 name='SequenceClassifier',
                 config={'threshold': 0.4})


data = pd.read_csv(DATA_FILE, sep='\t')
n = len(data)
labels = data['label']

predictions = []
for i, row in data.iterrows():
    predictions.append(SIMILARITY_CLASSIFIER.predict(row['out1'], row['out2']))

count = Counter(predictions)
negative = count[0] / n
positive = count[1] / n

accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)


wandb.log({'negative': negative, 'positive': positive, 'accuracy': accuracy,
           'precision': precision, 'recall': recall, 'f1': f1})
wandb.finish()
