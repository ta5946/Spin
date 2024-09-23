import pandas as pd

DATA_FILE = '../../data/out/llama/val.tsv'
OUT_FILE = '../../data/out/llama/mistakes.tsv'


data = pd.read_csv(DATA_FILE, sep='\t')

labels = data['label']
predictions = data['prediction']

mistakes = data[labels != predictions]
mistakes.to_csv(OUT_FILE, sep='\t', index=False)
