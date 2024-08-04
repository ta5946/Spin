import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = '../data/outcome_similarity/train.tsv'


data = pd.read_csv(DATA_FILE, sep='\t')
n = len(data)

data['label'].value_counts().plot(kind='bar')
plt.title('Similarity distribution for ' + str(n) + ' pairs of outcomes')
plt.xlabel('label')
plt.ylabel('count')
plt.show()
