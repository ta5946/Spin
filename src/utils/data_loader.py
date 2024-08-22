import pandas as pd


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_df(self):
        return pd.read_csv(self.data_path, sep='\t')

    def load_dict(self):
        data = self.load_df()
        data['label'] = data['label'].apply(lambda label: 'Yes' if label == 1 else 'No')
        return data.to_dict(orient='records')
