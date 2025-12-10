from preprocessing.data_loader import DataLoader
from preprocessing.data_cleaner import DataCleaner
from preprocessing.data_transformer import DataTransformer

class PreprocessingPipeline:
    def __init__(self):
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer(one_hot_encode=True, binarize_targets=True)

    def run(self):
        features, targets = self.loader.load_data()

        features = self.cleaner.clean_data(features)
        features, targets = self.transformer.transform(
            features, targets, categorical_columns_names=self.loader.categorical_columns
        )
        return features, targets

if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    features, targets = pipeline.run()

    print(features)
    print(targets)