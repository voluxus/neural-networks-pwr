import pandas as pd

class DataTransformer:
    def __init__(self, one_hot_encode=True, binarize_targets=False):
        self.one_hot_encode = one_hot_encode
        self.binarize_targets = binarize_targets

    def transform(self, features, targets, categorical_columns_names=None):
        if self.one_hot_encode:
            if categorical_columns_names is None:
                raise ValueError("Categorical column names must be provided for one-hot encoding.")
            features = self._one_hot_encode_transform(features, categorical_columns_names)

        if self.binarize_targets:
            targets = targets.loc[features.index].ne(0).astype(int)
        else:
            targets = targets.loc[features.index]

        return features, targets
    
    def _one_hot_encode_transform(self, data, categorical_column_names):
        """Applies one-hot encoding to the specified categorical columns."""

        return pd.get_dummies(data, columns=categorical_column_names)
    
    # TODO feature scaling

        