import pandas as pd
from pathlib import Path

from data_downloader import DataDownloader

class DataLoader:
    """Uses DataDownloader to load and provide the dataset as a DataFrame.
        But could be extended to include data cleaning and preprocessing steps in future.
    """
    def __init__(self, data_file='heart_disease.csv', categorical_columns=None):
        self.base_path = Path(__file__).parent / 'dataset'
        self.data_path = self.base_path / data_file
        
        self.data_downloader = DataDownloader()
        
        # Hardcoded categorical column names for heart disease dataset
        self.categorical_columns = categorical_columns or [
            'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'
        ]
        self.target_column = 'num'
    
    def get_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.categorical_columns]
    
    def get_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        exclude_cols = self.categorical_columns + [self.target_column]
        numerical_cols = [col for col in df.columns if col not in exclude_cols]
        return df[numerical_cols]
    
    def get_target_column(self, df: pd.DataFrame) -> pd.Series:
        return df[self.target_column]
    
    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        print(f"DataLoader: Loading data from {self.data_path}")
        df = self.data_downloader.get_dataset(self.data_path)
        
        targets = self.get_target_column(df)
        features = df.drop(columns=[self.target_column])
        
        return features, targets

if __name__ == "__main__":
    loader = DataLoader()
    data_frame = loader.load_data()
    print(data_frame)