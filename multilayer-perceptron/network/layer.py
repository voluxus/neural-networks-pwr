import numpy as np


class Layer:
    """Base class for neural network layers"""
    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def update_parameters(self, learning_rate):
        pass  # Not all layers have parameters


class DenseLayer(Layer):
    """Fully connected layer"""
    def __init__(self, input_size: int, output_size: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)  # He initialization
        self.bias = np.zeros((1, output_size))  # A bias for each neuron
    
        self.cache_input = None  # Cache input matrix (batch) - for backward pass
        
        self.grad_weights = None  # Gradient of error function with respect to weights, calculated in backward
        self.grad_bias = None  # Gradient of error function with respect to bias, calculated in backward
    
    def forward(self, X):
        """
        Forward pass: z = X @ W + b
        X: (batch_size, input_size)
        Returns: (batch_size, output_size)
        """
        self.cache_input = X
        return X @ self.weights + self.bias
    
    def backward(self, grad_output):
        """
        Backward pass
        grad_output: (batch_size, output_size) - gradient from next layer
        Returns: (batch_size, input_size) - gradient to pass to previous layer


        Honestly, the most mind-bending part is the shapes, arbitrarily chosen matrix order and the averaging over the batch
        """
        batch_size = self.cache_input.shape[0]
        
        # Gradient with respect to weights: X^T @ grad_output
        self.grad_weights = (self.cache_input.T @ grad_output) / batch_size  # average over batch (@ produces a sum over batch_size in fact)
        
        # Gradient with respect to bias: average over batch
        self.grad_bias = np.mean(grad_output, axis=0, keepdims=True)
        
        # Gradient with respect to input: grad_output @ W^T
        grad_input = grad_output @ self.weights.T
        
        return grad_input
    
    def update_parameters(self, learning_rate):
        """Update weights and bias using gradients"""
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias


class SigmoidActivation(Layer):
    """Sigmoid activation function"""
    def __init__(self):
        self.cache_output = None
    
    def forward(self, X):
        """
        Forward pass: sigmoid(X)
        X: (batch_size, features)
        """
        self.cache_output = self._sigmoid(X)
        return self.cache_output
    
    def backward(self, grad_output):
        """
        Backward pass
        Derivative of sigmoid: σ(x) * (1 - σ(x))
        """
        sigmoid_derivative = self.cache_output * (1 - self.cache_output)
        return grad_output * sigmoid_derivative
    
    def _sigmoid(self, z):
        """Numerically stable sigmoid"""
        positive_mask = (z >= 0)
        negative_mask = ~positive_mask
        
        output = np.empty_like(z)
        output[positive_mask] = 1 / (1 + np.exp(-z[positive_mask]))
        
        exp_z = np.exp(z[negative_mask])
        output[negative_mask] = exp_z / (1 + exp_z)
        
        return output


class ReLUActivation(Layer):
    """ReLU activation function"""
    def __init__(self):
        self.cache_input = None
    
    def forward(self, X):
        """Forward pass: max(0, X)"""
        self.cache_input = X
        return np.maximum(0, X)
    
    def backward(self, grad_output):
        """Backward pass: gradient is 1 where X > 0, else 0"""
        grad_input = grad_output.copy()
        grad_input[self.cache_input <= 0] = 0
        return grad_input


class TanhActivation(Layer):
    """Tanh activation function"""
    def __init__(self):
        self.cache_output = None
    
    def forward(self, X):
        """Forward pass: tanh(X)"""
        self.cache_output = np.tanh(X)
        return self.cache_output
    
    def backward(self, grad_output):
        """Backward pass: 1 - tanh²(X)"""
        tanh_derivative = 1 - self.cache_output ** 2
        return grad_output * tanh_derivative