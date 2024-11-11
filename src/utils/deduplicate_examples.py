import pandas as pd

DATA_FILE = '../../data/outcome_similarity/test.tsv'


data = pd.read_csv(DATA_FILE, sep='\t')
clean_data = data.drop_duplicates(subset=['out1', 'out2', 'label'])

n_duplicates = len(data) - len(clean_data)
clean_data.to_csv(DATA_FILE, sep='\t', index=False)

print('Cleaned ' + str(n_duplicates) + ' duplicate rows from ' + DATA_FILE)
