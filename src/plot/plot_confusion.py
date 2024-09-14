import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

DATA_FILE = '../../data/out/llama/val.tsv'
MODEL_NAME = 'Llama'
MODEL_THRESHOLD = 0.5


data = pd.read_csv(DATA_FILE, sep='\t')
n_rows = len(data)

scores = data['score']
labels = data['label']
predictions = [1 if score > MODEL_THRESHOLD else 0 for score in scores]

print('Accuracy: ' + str(accuracy_score(labels, predictions)))

confusion_matrix = pd.crosstab(labels, predictions, rownames=['actual'], colnames=['predicted'])
sns.heatmap(confusion_matrix, cmap='Reds', annot=True, fmt='d')
plt.title(MODEL_NAME + ' confusion matrix for outcome similarity')
plt.show()
