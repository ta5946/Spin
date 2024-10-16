import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = '../../data/out/llama/fix.tsv'
MODEL_NAME = 'Llama'
MODEL_THRESHOLD = 0.5


data = pd.read_csv(DATA_FILE, sep='\t')
n_rows = len(data)

scores = data['score']
labels = data['label']
colors = ['blue' if label == 1 else 'red' for label in labels]
positions = np.random.uniform(0, 1, n_rows)

plt.figure()
plt.scatter(scores, positions, c=colors, edgecolors='black', alpha=0.7)
plt.axvline(x=MODEL_THRESHOLD, color='gray', linestyle='--')
plt.yticks([])
plt.xlabel('score')
plt.title(MODEL_NAME + ' similarity score scatter for ' + str(n_rows) + ' pairs of outcomes')
plt.show()
