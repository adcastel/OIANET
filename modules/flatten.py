from modules.layer import Layer

class Flatten(Layer):
    def forward(self, input):  # input: [batch][channels][height][width]
        self.input_shape = (len(input), len(input[0]), len(input[0][0]), len(input[0][0][0]))
        return [
            [pixel for channel in sample for row in channel for pixel in row]
            for sample in input
        ]

    def backward(self, grad_output, learning_rate=None):
        batch, channels, height, width = self.input_shape
        grad_input = []

        for b in range(batch):
            sample = []
            offset = 0
            for c in range(channels):
                channel = []
                for h in range(height):
                    row = []
                    for w in range(width):
                        row.append(grad_output[b][offset])
                        offset += 1
                    channel.append(row)
                sample.append(channel)
            grad_input.append(sample)

        return grad_input

import numpy as np

class Flatten_np(Layer):
    def forward(self, input):  # input: np.ndarray of shape [B, C, H, W]
        self.input_shape = input.shape  # Save shape for backward
        return input.reshape(input.shape[0], -1)  # Flatten each sample in batch

    def backward(self, grad_output, learning_rate=None):
        return grad_output.reshape(self.input_shape)  # Reshape back to original 4D