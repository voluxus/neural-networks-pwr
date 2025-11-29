import data_loader
import data_cleaner
import data_transformer

class PreprocessingPipeline:
    def __init__(self):
        self.loader = data_loader.DataLoader()
        self.cleaner = data_cleaner.DataCleaner()
        self.transformer = data_transformer.DataTransformer(one_hot_encode=True, binarize_targets=True)

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