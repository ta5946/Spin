import pandas as pd


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load(self):
        return pd.read_csv(self.data_path, sep='\t')
