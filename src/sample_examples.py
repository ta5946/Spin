import pandas as pd
import os

DATA_FILE = '../data/outcome_similarity/val.tsv'
OUT_FILE = '../data/outcome_similarity/train.tsv'


data = pd.read_csv(DATA_FILE, sep='\t')
n_rows = len(data)

if n_rows < 100:
    print('Not enough rows to sample')
    exit()

examples = data.sample(n=100, random_state=0)
examples.to_csv(OUT_FILE, sep='\t', index=False)

data = data.drop(examples.index)
data.to_csv(DATA_FILE, sep='\t', index=False)

print('Sampled 100 rows from ' + os.path.basename(DATA_FILE) + ' to ' + os.path.basename(OUT_FILE))
