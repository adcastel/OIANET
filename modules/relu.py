from modules.layer import Layer

# class ReLU:
#     def __init__(self):
#         self.input = None  # To store input during forward pass
    
#     def forward(self, input):
#         # Save input for backward pass
#         self.input = input
#         output = []

#         # Apply ReLU element-wise using loops for a 4D tensor (batch_size, height, width, channels)
#         for batch in input:
#             batch_output = []
#             for row in batch:
#                 row_output = []
#                 for pixel in row:
#                     if isinstance(pixel, list):
#                         # Apply ReLU to each pixel in each row (for multi-channel data)
#                         pixel_output = []
#                         for x in pixel:
#                             pixel_output.append(max(0, x))  # Apply ReLU to each element in the pixel (channel)
#                         row_output.append(pixel_output)
#                     else:
#                         print("Aqui")
#                         pixel_output.append(max(0, pixel))
#                         row_output.append(pixel_output)
#                 batch_output.append(row_output)
#             output.append(batch_output)

#         return output
    
#     def backward(self, grad_output):
#         grad_input = []

#         # Propagate gradients: where input > 0, pass the gradient; else, pass 0
#         for i, batch in enumerate(self.input):
#             grad_batch = []
#             for j, row in enumerate(batch):
#                 grad_row = []
#                 for k, pixel in enumerate(row):
#                     grad_pixel = []
#                     for l, x in enumerate(pixel):
#                         # Gradient is grad_output if input > 0, else 0
#                         grad_pixel.append(grad_output[i][j][k][l] if x > 0 else 0)
#                     grad_row.append(grad_pixel)
#                 grad_batch.append(grad_row)
#             grad_input.append(grad_batch)

#         return grad_input

class ReLU(Layer):
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return self._relu(input)

    def backward(self, grad_output,learning_rate):
        return self._relu_backward(self.input, grad_output)

    def _relu(self, data):
        if isinstance(data, (int, float)):
            return max(0, data)
        return [self._relu(x) for x in data]

    def _relu_backward(self, input_data, grad_output):
        if isinstance(input_data, (int, float)):
            return grad_output if input_data > 0 else 0
        return [self._relu_backward(input_data[i], grad_output[i]) for i in range(len(input_data))]

import numpy as np

class ReLU_np(Layer):
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output, learning_rate):
        return grad_output * (self.input > 0)