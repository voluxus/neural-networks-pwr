from pathlib import Path
from ucimlrepo import fetch_ucirepo
import os
import pandas as pd

class DataDownloader:
    """Wrapper around UCI ML Repository fetcher to download and save the Heart Disease dataset locally."""
    def __init__(self):
        self.ucimlrepo_name="Heart Disease"

    def check_dataset_exists(self, dataset_path):
        return os.path.exists(dataset_path)

    def download_dataset(self):
        dataset = fetch_ucirepo(name=self.ucimlrepo_name)
        return dataset

    def save_dataset(self, dataset, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        
        df.to_csv(filepath, index=False)
        print(f"DataDownloader: Dataset saved to {filepath}")

    def get_dataset(self, filepath=Path(__file__).parent / 'dataset/heart_disease.csv'):
        if self.check_dataset_exists(filepath):
            print(f"DataDownloader: Loading existing dataset from {filepath}")
            return pd.read_csv(filepath)
        else:
            print(f"DataDownloader: Dataset not found. Downloading from UCI ML Repository...")
            dataset = self.download_dataset()
            self.save_dataset(dataset, filepath)
            return pd.read_csv(filepath)

if __name__ == "__main__":
    downloader = DataDownloader()
    data_frame = downloader.get_dataset()
    print(data_frame)