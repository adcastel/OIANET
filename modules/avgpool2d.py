import numpy as np
from modules.layer import Layer

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
