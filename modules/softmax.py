import math
from modules.layer import Layer

class Softmax(Layer):
    def forward(self, input):
        self.output = []
        for row in input:
            max_val = max(row)
            exps = [math.exp(x - max_val) for x in row]
            sum_exps = sum(exps)
            self.output.append([x / sum_exps for x in exps])
        return self.output

    def backward(self, grad_output, learning_rate):
        # Assuming softmax used with cross-entropy loss
        return grad_output

import numpy as np

class Softmax_np(Layer):
    def forward(self, input):  # input: [batch_size x num_classes]
        input = np.array(input).astype(dtype=np.float32)  # Ensure input is float for numerical stability
        self.output = np.zeros_like(input,np.float32)

        for i, row in enumerate(input):
            max_val = np.max(row)
            exps = np.exp(row - max_val)
            self.output[i] = exps / np.sum(exps)

        return self.output

    def backward(self, grad_output, learning_rate=None):
        # Assuming softmax used with cross-entropy loss, so gradient is simplified
        return grad_output