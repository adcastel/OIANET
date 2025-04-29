import numpy as np
from modules.layer import Layer

"""
class GlobalAvgPool2D(Layer):
    def forward(self, x):  # shape: [batch, channels, h, w]
        self.input = x
        batch_size, channels, h, w = x.shape
        return np.array([
            [
                np.mean(x[b, c]) for c in range(channels)
            ] for b in range(batch_size)
        ], dtype=np.float32)

    def backward(self, grad_output, learning_rate=None):
        batch_size, channels, h, w = self.input.shape
        grad_input = np.zeros_like(self.input)
        for b in range(batch_size):
            for c in range(channels):
                grad = grad_output[b][c] / (h * w)
                grad_input[b, c] = grad
        return grad_input
"""
class GlobalAvgPool2D(Layer):

    def __init__(self):
        self.input = None
        
    def forward(self, x):  # shape: [batch, channels, h, w]
        self.input = x
        return np.mean(x, axis=(2, 3), keepdims=False).astype(np.float32)  # shape: [batch, channels]

    def backward(self, grad_output, learning_rate=None):
        batch_size, channels, h, w = self.input.shape
        grad = grad_output[:, :, None, None] / (h * w)  # shape: [batch, channels, 1, 1]
        return np.ones_like(self.input) * grad  # broadcast to [batch, channels, h, w]