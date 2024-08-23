import pandas as pd


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_df(self):
        return pd.read_csv(self.data_path, sep='\t')

    def load_dict(self):
        data = self.load_df()
        data['label'] = data['label'].apply(lambda label: 'Yes, the primary outcome is correctly reported.' if label == 1 else 'No, the primary outcome is not correctly reported.')
        return data.to_dict(orient='records')
