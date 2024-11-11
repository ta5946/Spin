import pandas as pd
import os

DATA_FILE = '../data/outcome_similarity/test.tsv'
OUT_FILE = '../data/outcome_similarity/dev.tsv'
N_EXAMPLES = 200

data = pd.read_csv(DATA_FILE, sep='\t')
n_rows = len(data)

if n_rows < N_EXAMPLES:
    print('Not enough rows to sample')
    exit()

examples = data.sample(n=N_EXAMPLES, random_state=0)
examples.to_csv(OUT_FILE, sep='\t', index=False)

data = data.drop(examples.index)
data.to_csv(DATA_FILE, sep='\t', index=False)

print('Sampled ' + str(N_EXAMPLES) + ' rows from ' + os.path.basename(DATA_FILE) + ' to ' + os.path.basename(OUT_FILE))
