import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

DATA_DIR = '../../data/out/'
DATA_FILES = ['stems/val.tsv', 'olmo/val.tsv', 'mistral/val.tsv', 'bio/val.tsv', 'llama/val.tsv']
MODEL_NAMES = ['Stems', 'Olmo', 'Mistral', 'BioMistral', 'Llama']
MODEL_COLORS = ['red', 'yellow', 'orange', 'green', 'blue']


plt.figure()
for i in range(len(DATA_FILES)):
    data = pd.read_csv(DATA_DIR + DATA_FILES[i], sep='\t')
    fpr, tpr, thresholds = roc_curve(data['label'], data['score'])
    plt.plot(fpr, tpr, color=MODEL_COLORS[i], label=MODEL_NAMES[i])

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver operating characteristic curves for similarity models')
plt.legend()
plt.show()
