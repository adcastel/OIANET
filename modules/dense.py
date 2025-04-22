from modules.utils import matmul, vector_add, transpose
import random
from modules.layer import Layer
"""
class Dense:
    def __init__(self, in_features, out_features):
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(out_features)] for _ in range(in_features)]
        self.biases = [0.0 for _ in range(out_features)]
        self.input = None

    def forward(self, input):
        
        self.input = input  # [batch_size][in_features]
        
        out = matmul(input, self.weights)  # [batch_size x out_features]
        self.output = vector_add(out, [self.biases for _ in range(len(input))])
        return self.output
"""
class Dense(Layer):
    def __init__(self, in_features, out_features):
        self.weights = [[random.gauss(0, 1 / in_features**0.5) for _ in range(out_features)] for _ in range(in_features)]
        self.biases = [0 for _ in range(out_features)]
        self.input = None

    def forward(self, input):
        self.input = input  # input is [batch_size x in_features]
        #print(f"Input shape: {len(input)} x {len(input[0])}")
        #print(input)
        #print(f"Weights shape: {len(self.weights)} x {len(self.weights[0])}")
        self.output = matmul(input, self.weights)
        for i in range(len(self.output)):
            for j in range(len(self.output[0])):
                self.output[i][j] += self.biases[j]
        return self.output

    def backward(self, grad_output, learning_rate):
        # grad_output: [batch_size x out_features]
        input_T = transpose(self.input)  # [in_features x batch_size]
        grad_weights = matmul(input_T, grad_output)  # [in_features x out_features]

        grad_biases = [0 for _ in self.biases]
        for row in grad_output:
            for i, val in enumerate(row):
                grad_biases[i] += val

        grad_input = matmul(grad_output, transpose(self.weights))  # [batch_size x in_features]

        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.weights[i][j] -= learning_rate * grad_weights[i][j]

        for i in range(len(self.biases)):
            self.biases[i] -= learning_rate * grad_biases[i]

        return grad_input
    

import numpy as np

class Dense_np(Layer):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(in_features, out_features).astype(np.float32) * (1 / in_features**0.5)
        self.biases = np.zeros(out_features,dtype=np.float32)
        self.input = None

    def forward(self, input):  # input: [batch_size x in_features]
        self.input = np.array(input).astype(np.float32)  # Ensure input is float for numerical stability
        batch_size = self.input.shape[0]

        output = np.zeros((batch_size, self.out_features),dtype=np.float32)
        for i in range(batch_size):
            for j in range(self.out_features):
                for k in range(self.in_features):
                    output[i][j] += self.input[i][k] * self.weights[k][j]
                output[i][j] += self.biases[j]

        self.output = output
        return output

    def backward(self, grad_output, learning_rate):
        grad_output = np.array(grad_output).astype(np.float32)  # Ensure grad_output is float for numerical stability
        batch_size = grad_output.shape[0]

        # Gradient w.r.t. weights
        grad_weights = np.zeros((self.in_features, self.out_features),dtype=np.float32)
        for i in range(self.in_features):
            for j in range(self.out_features):
                for b in range(batch_size):
                    grad_weights[i][j] += self.input[b][i] * grad_output[b][j]

        # Gradient w.r.t. biases
        grad_biases = np.sum(grad_output, axis=0)

        # Gradient w.r.t. input
        grad_input = np.zeros((batch_size, self.in_features),dtype=np.float32)
        for b in range(batch_size):
            for i in range(self.in_features):
                for j in range(self.out_features):
                    grad_input[b][i] += grad_output[b][j] * self.weights[i][j]

        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input
