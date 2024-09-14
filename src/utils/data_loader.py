import pandas as pd


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_df(self):
        return pd.read_csv(self.data_path, sep='\t')

    def load_dict(self, str_only=False):
        data = self.load_df()
        data['answer'] = data['label'].apply(lambda label: 'No, the primary outcome is not correctly reported.' if label == 0 else 'Yes, the primary outcome is correctly reported.')
        if str_only:
            data = data.select_dtypes(include='object')
        return data.to_dict(orient='records')
