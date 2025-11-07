"""
    Loading and preprocessing
"""
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import pandas as pd

heart_disease = fetch_ucirepo(name='Heart Disease')

features = heart_disease.data.features
targets = heart_disease.data.targets

# Clean data, really tiny margin of NAs checked in EDA
features = features.dropna()

# One-hot encoding
categorical_indexes = [i for i, t in heart_disease.variables.type.items() if t in ['Categorical']]
categorical_features = heart_disease.data.features.iloc[:, categorical_indexes]
X_processed = pd.get_dummies(features, columns=categorical_features.columns)

print(X_processed)

# Output classes transformed into binary: 0 or 1 of illness presence
y_aligned = targets.loc[X_processed.index, "num"].ne(0).astype(int)
# .ne(0) is (num != 0) -> 0/1, this transforms sick classes (1,2,3,4) into single (1)

print(y_aligned)

"""
    The Model
"""

import numpy as np


class LogisticRegressionModel:
    def __init__(self,
                 seed: int = None,
                 weights: np.ndarray = None,
                 bias: float = None,
                 learning_rate: float = 0.1
                 ):

        if seed is not None:
            np.random.seed(seed)

        self.weights = None if weights is None else weights.copy()
        self.bias = np.random.randn() if bias is None else bias
        self.learning_rate = learning_rate

    def fit(self, X, y, epochs=1, batch_size=32):
        X, y = self._convert_data_to_numpy(X, y)
        N, D = X.shape

        if self.weights is None:
            self._initialize_weights(D)

        self._fit_numpy(X, y, epochs=epochs, batch_size=batch_size)

    def _convert_data_to_numpy(self, X, y):
        if hasattr(X, "to_numpy"):
            X_np = X.to_numpy(dtype=np.float32, copy=False)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        if hasattr(y, "to_numpy"):
            y_np = y.to_numpy(dtype=np.float32, copy=False)
        else:
            y_np = np.asarray(y, dtype=np.float32)

        return X_np, y_np  # will it be immutable tuple and thus force later copy to modify?

    def _initialize_weights(self, feature_count: int):
        self.weights = np.random.randn(feature_count)

    def _fit_numpy(self, X, y, epochs, batch_size):
        N, D = X.shape
        eps = 1e-15  # to clip for numerical stability

        for epoch in range(epochs):
            # shuffle
            shuffled_indices = np.random.permutation(N)  # (N,)
            X = X[shuffled_indices]
            y = y[shuffled_indices]

            # mini-batches
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)  # exclusive index
                X_batch = X[start:end]  # shape (m, D); m- the true batch size
                y_batch = y[start:end]  # shape (m,)

                m = X_batch.shape[0]  # m can be < batch_size if the sequence is about to end

                # forward
                z = X_batch @ self.weights + self.bias  # (m,)
                y_pred = self._sigmoid(z)  # (m,)
                #print(y_pred, "\n")

                # gradient of logistic loss: grad_w = X_batch.T @ (y_pred - y) / m
                error = y_pred - y_batch  # (m,)
                grad_w = (X_batch.T @ error) / m  # (D,)
                grad_b = np.mean(error)  # scalar

                #print("grad_w", grad_w)
                #print("grad_b", grad_b)
                #print("\n")

                # update parameters - average loss gradient value per batch
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            # end of epoch - compute loss on full dataset for logging
            z_dataset = X @ self.weights + self.bias  # (N,)
            y_pred_dataset_clipped = np.clip(self._sigmoid(z_dataset), eps, 1 - eps)
            loss_epoch = np.mean(-y * np.log(y_pred_dataset_clipped) - (1 - y) * np.log(1 - y_pred_dataset_clipped))

            print(f"Epoch {epoch + 1}/{epochs} — loss: {loss_epoch:.6f}", flush=True)

    def predict(self, X_batch: np.ndarray) -> np.ndarray:
        z = X_batch @ self.weights + self.bias  # (m,)
        y_pred = self._sigmoid(z)  # (m,)
        return y_pred

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        The sigmoid has to be numerically stable.

        Currently input data is not normalized, so the z sum gets big (-) or (+) numbers.
        The np.exp(-z) produces 0 on big positive z, but if z is big (-), the temporary result is
        a very big number - not possible to be represented; so np returns +inf which is inaccurate
        and then the whole sigmoid expression has 1 / +inf which outputs 0.0
        So the model takes radical step to the weights, each update jittering between 0.0 and 1.0
        And the model is unable to learn
        """
        # z: (n,) - vectorized operation
        positive_mask = (z >= 0)
        negative_mask = ~positive_mask

        output_batch = np.empty_like(z)

        output_batch[positive_mask] = 1 / (1 + np.exp(-z[positive_mask]))

        exp_z = np.exp(z[negative_mask])
        output_batch[negative_mask] = exp_z / (1 + exp_z)

        return output_batch


"""
    Learning and evaluation
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def evaluate_binary(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_pred": y_pred
    }

X_np = X_processed.to_numpy(dtype=np.float32)
y_np = y_aligned.to_numpy(dtype=np.int32)

#RANDOM_STATE = np.random.randint(100)
#RANDOM_STATE = 14
#RANDOM_STATE = 75
#RANDOM_STATE = 6
RANDOM_STATE=42

X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=RANDOM_STATE, stratify=y_np
)

scaler = StandardScaler()
scaler.fit(X_train)  # learn mean and stddev from train dataset

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = LogisticRegressionModel(seed=None, learning_rate=0.1)
classifier.fit(X_train, y_train, epochs=150, batch_size=32)

# After training:
y_test_probs = classifier.predict(X_test)
eval_res = evaluate_binary(y_test, y_test_probs, threshold=0.5)

print("\nTest set metrics:")
print(f"Accuracy:  {eval_res['accuracy']:.4f}")
print(f"Precision: {eval_res['precision']:.4f}")
print(f"Recall:    {eval_res['recall']:.4f}")
print(f"F1:        {eval_res['f1']:.4f}\n")

print(f"\nRandom state: {RANDOM_STATE}")