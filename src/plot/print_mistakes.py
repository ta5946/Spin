import pandas as pd

DATA_FILE = '../../data/out/llama/val.tsv'
OUT_FILE = '../../data/out/llama/mistakes.tsv'
MODEL_THRESHOLD = 0.1


data = pd.read_csv(DATA_FILE, sep='\t')

scores = data['score']
labels = data['label']
predictions = [1 if score > MODEL_THRESHOLD else 0 for score in scores]

mistakes = data[labels != predictions]
mistakes.to_csv(OUT_FILE, sep='\t', index=False)
