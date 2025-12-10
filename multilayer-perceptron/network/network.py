import numpy as np

from network.layer import DenseLayer, ReLUActivation, SigmoidActivation, TanhActivation

class Network:
    def __init__(
        self,
        hidden_layers: list[int],
        input_size: int,
        output_size: int,
        activation: str = 'sigmoid',
        multiclass: bool = False,
        learning_rate: float = 0.1,
        weight_init_std: float = None,
        seed: int = None
    ):
        """
        Initialize MLP network
        
        Args:
            hidden_layers: List of hidden layer sizes, e.g., [64, 32] for 2 hidden layers
            input_size: Number of input features
            output_size: Number of output classes (1 for binary, >1 for multiclass)
            activation: Activation function ('sigmoid', 'relu', 'tanh')
            multiclass: If True, use softmax + cross-entropy; if False, use sigmoid + binary cross-entropy
            learning_rate: Learning rate for gradient descent
            weight_init_std: Standard deviation for weight initialization (overrides default)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.multiclass = multiclass
        self.learning_rate = learning_rate
        self.weight_init_std = weight_init_std
        self.output_size = output_size
        self.layers = []
        
        # Build network architecture
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # Create hidden layers with activation
        for i in range(len(layer_sizes) - 1):
            # Dense layer
            dense = DenseLayer(layer_sizes[i], layer_sizes[i + 1], seed=seed)
            
            # Apply custom weight initialization if specified
            if weight_init_std is not None:
                dense.weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * weight_init_std
            
            self.layers.append(dense)
            
            # Activation layer (not for output layer - handled separately)
            if i < len(layer_sizes) - 2:
                self.layers.append(self._get_activation(activation))
        
        # Output layer activation (sigmoid for binary, softmax for multiclass)
        if multiclass:
            # Softmax will be applied in loss computation for numerical stability
            pass
        else:
            self.layers.append(SigmoidActivation())
    
    def _get_activation(self, activation: str):
        """Get activation layer by name"""
        activations = {
            'sigmoid': SigmoidActivation,
            'relu': ReLUActivation,
            'tanh': TanhActivation
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(activations.keys())}")
        
        return activations[activation]()
    
    def forward(self, X):
        """Forward pass through all layers"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad_output):
        """Backward pass through all layers"""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def update_parameters(self):
        """Update parameters in all layers"""
        for layer in self.layers:
            layer.update_parameters(self.learning_rate)
    
    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        Train the network
        
        Args:
            X: Training features (N, input_size)
            y: Training targets (N,) for binary or (N, output_size) for multiclass
            epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Print training progress
        """
        X, y = self._convert_data_to_numpy(X, y)
        N = X.shape[0]
        
        # Prepare targets for multiclass
        if self.multiclass and y.ndim == 1:
            # Validate first
            n_classes = len(np.unique(y))
            if n_classes != self.output_size:
                raise ValueError(f"Expected {self.output_size} classes, got {n_classes}")
            
            # One-hot encode the output labels
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y.astype(int)] = 1
            y = y_one_hot
        elif self.multiclass and y.ndim == 2:
            # Already one-hot encoded - just validate shape
            if y.shape[1] != self.output_size:
                raise ValueError(f"Expected {self.output_size} output classes, got {y.shape[1]}")
        elif not self.multiclass and y.ndim == 1:
            # Already correct format for binary
            pass
        else:
            raise ValueError(f"Incompatible target shape {y.shape} for multiclass={self.multiclass}")
        
        # Prepare targets for multiclass
        if self.multiclass and y.ndim == 1:
            # One-hot encode if needed
            n_classes = len(np.unique(y))
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y.astype(int)] = 1
            y = y_one_hot
        elif not self.multiclass and y.ndim == 2:
            # Flatten for binary classification
            y = y.ravel()
        
        if (self.multiclass and y.ndim == 1) or (not self.multiclass and y.ndim == 2):
            raise ValueError("Target shape is not compatible with the classification type.")
        
        eps = 1e-15  # For numerical stability
        
        for epoch in range(epochs):
            # Shuffle data
            shuffled_indices = np.random.permutation(N)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            
            # Mini-batch training
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute gradient of loss
                if self.multiclass:
                    # Softmax + cross-entropy gradient: y_pred - y_true
                    # Apply softmax here for numerical stability
                    y_pred_softmax = self._softmax(y_pred)
                    grad_loss = (y_pred_softmax - y_batch) / X_batch.shape[0]
                else:
                    # Binary cross-entropy gradient
                    grad_loss = (y_pred - y_batch.reshape(-1, 1)) / X_batch.shape[0]
                
                # Backward pass
                self.backward(grad_loss)
                
                # Update parameters
                self.update_parameters()
            
            # Calculate and print loss for the epoch
            if verbose:
                y_pred_full = self.forward(X)
                
                if self.multiclass:
                    y_pred_full = self._softmax(y_pred_full)
                    y_pred_clipped = np.clip(y_pred_full, eps, 1 - eps)
                    loss = -np.mean(np.sum(y * np.log(y_pred_clipped), axis=1))
                else:
                    y_pred_clipped = np.clip(y_pred_full.ravel(), eps, 1 - eps)
                    loss = -np.mean(
                        y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped)
                    )
                
                print(f"Epoch {epoch + 1}/{epochs} — loss: {loss:.6f}", flush=True)
    
    def predict(self, X):
        """
        Make predictions
        
        Returns:
            For binary: probabilities (N,)
            For multiclass: class probabilities (N, n_classes)
        """
        X = self._convert_data_to_numpy(X)[0]
        y_pred = self.forward(X)
        
        if self.multiclass:
            return self._softmax(y_pred)
        else:
            return y_pred.ravel()
    
    def predict_classes(self, X):
        """
        Predict class labels
        
        Returns:
            For binary: 0 or 1 (N,)
            For multiclass: class index (N,)
        """
        probabilities = self.predict(X)
        
        if self.multiclass:
            return np.argmax(probabilities, axis=1)  # go along rows (for each row, pick max)
        else:
            return (probabilities >= 0.5).astype(int)
    
    def _softmax(self, z):
        """Numerically stable softmax"""
        # Subtract max for numerical stability; 
        # e^(z - C) is identical in relative terms to e^z (when divided by the sum); 
        # when C = max(z), makes largest exp value 1, others <1, avoids overflow
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _convert_data_to_numpy(self, X, y=None):
        """Convert data to numpy arrays"""
        if hasattr(X, "to_numpy"):
            X_np = X.to_numpy(dtype=np.float32, copy=False)
        else:
            X_np = np.asarray(X, dtype=np.float32)
        
        if y is not None:
            if hasattr(y, "to_numpy"):
                y_np = y.to_numpy(dtype=np.float32, copy=False)
            else:
                y_np = np.asarray(y, dtype=np.float32)
            return X_np, y_np
        
        return X_np, None