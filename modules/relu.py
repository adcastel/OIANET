from modules.layer import Layer



import numpy as np

class ReLU(Layer):
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output, learning_rate):
        return grad_output * (self.input > 0)