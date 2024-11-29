import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

OLMO_FILE = '../../data/test/olmo_detail/predictions.tsv'
BIO_FILE = '../../data/test/bio_detail/predictions.tsv'
LLAMA_FILE = '../../data/test/mistral_detail/predictions.tsv'
VOTE_THRESHOLD = 2


olmo_data = pd.read_csv(OLMO_FILE, sep='\t')
bio_data = pd.read_csv(BIO_FILE, sep='\t')
llama_data = pd.read_csv(LLAMA_FILE, sep='\t')

labels = olmo_data['label']
olmo_predictions = olmo_data['prediction']
bio_predictions = bio_data['prediction']
llama_predictions = llama_data['prediction']

votes = olmo_predictions + bio_predictions + llama_predictions
predictions = [1 if vote >= VOTE_THRESHOLD else 0 for vote in votes]

metrics = {
    'negative_ratio': predictions.count(0) / len(predictions),
    'positive_ratio': predictions.count(1) / len(predictions),
    'accuracy_score': accuracy_score(labels, predictions),
    'precision_score': precision_score(labels, predictions),
    'recall_score': recall_score(labels, predictions),
    'f1_score': f1_score(labels, predictions),
    'auc_score': roc_auc_score(labels, predictions)
}
print(metrics)
