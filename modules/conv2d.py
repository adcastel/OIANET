from modules.layer import Layer
import random

def pad2d(input, pad):
    h, w = len(input), len(input[0])
    padded = [[0] * (w + 2 * pad) for _ in range(h + 2 * pad)]
    for i in range(h):
        for j in range(w):
            padded[i + pad][j + pad] = input[i][j]
    return padded

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernels = [
            [  # One kernel per input channel
                [[random.uniform(-0.1, 0.1) for _ in range(kernel_size)] for _ in range(kernel_size)]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        self.biases = [0.0 for _ in range(out_channels)]

    def forward(self, input):  # input: [batch][in_channels][height][width]
        self.input = input
        batch_size = len(input)
        self.output = []

        for b in range(batch_size):
            sample = input[b]
            out_maps = []

            for out_c in range(self.out_channels):
                # Initialize output channel accumulator
                out_channel = None

                for in_c in range(self.in_channels):
                    # Get kernel and input
                    kernel = self.kernels[out_c][in_c]
                    input_ch = sample[in_c]

                    # Apply padding if needed
                    if self.padding > 0:
                        in_h, in_w = len(input_ch), len(input_ch[0])
                        padded = [[0] * (in_w + 2 * self.padding) for _ in range(in_h + 2 * self.padding)]
                        for i in range(in_h):
                            for j in range(in_w):
                                padded[i + self.padding][j + self.padding] = input_ch[i][j]
                        input_ch = padded

                    # Get dimensions
                    in_h, in_w = len(input_ch), len(input_ch[0])
                    k_h, k_w = len(kernel), len(kernel[0])
                    out_h = (in_h - k_h) // self.stride + 1
                    out_w = (in_w - k_w) // self.stride + 1

                    # Convolve input with kernel
                    conv_out = [[0 for _ in range(out_w)] for _ in range(out_h)]
                    for i in range(out_h):
                        for j in range(out_w):
                            acc = 0
                            for ki in range(k_h):
                                for kj in range(k_w):
                                    acc += input_ch[i * self.stride + ki][j * self.stride + kj] * kernel[ki][kj]
                            conv_out[i][j] = acc

                    # Accumulate over input channels
                    if out_channel is None:
                        out_channel = conv_out
                    else:
                        for i in range(out_h):
                            for j in range(out_w):
                                out_channel[i][j] += conv_out[i][j]

                # Add bias
                for i in range(len(out_channel)):
                    for j in range(len(out_channel[0])):
                        out_channel[i][j] += self.biases[out_c]

                out_maps.append(out_channel)
            self.output.append(out_maps)
        return self.output

    def backward(self, grad_output, learning_rate):
        batch_size = len(self.input)
        grad_input = [[[  # Zero gradients for each input
            [0] * len(self.input[0][0][0])
            for _ in range(len(self.input[0][0]))
        ] for _ in range(self.in_channels)] for _ in range(batch_size)]

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    kernel = self.kernels[out_c][in_c]
                    k_h, k_w = len(kernel), len(kernel[0])
                    input_ch = self.input[b][in_c]

                    # Apply padding to input if needed
                    in_h, in_w = len(input_ch), len(input_ch[0])
                    padded = [[0] * (in_w + 2 * self.padding) for _ in range(in_h + 2 * self.padding)]
                    for i in range(in_h):
                        for j in range(in_w):
                            padded[i + self.padding][j + self.padding] = input_ch[i][j]
                    
                    in_h_p, in_w_p = len(padded), len(padded[0])
                    grad = grad_output[b][out_c]

                    # Update kernel weights
                    for i in range(len(grad)):
                        for j in range(len(grad[0])):
                            for ki in range(k_h):
                                for kj in range(k_w):
                                    #r = i * self.stride + ki
                                    #c = j * self.stride + kj
                                    #self.kernels[out_c][in_c][ki][kj] -= learning_rate * input_ch[r][c] * grad[i][j]
                                    #grad_input[b][in_c][r][c] += kernel[ki][kj] * grad[i][j]
                                    r = i * self.stride + ki
                                    c = j * self.stride + kj
                                    if 0 <= r < in_h_p and 0 <= c < in_w_p:
                                        val = padded[r][c]
                                        # Update kernel
                                        self.kernels[out_c][in_c][ki][kj] -= learning_rate * val * grad[i][j]

                                        # Compute gradients for input (remove padding when writing)
                                        if self.padding <= r < in_h_p - self.padding and self.padding <= c < in_w_p - self.padding:
                                            grad_input[b][in_c][r - self.padding][c - self.padding] += kernel[ki][kj] * grad[i][j]

                # Update biases
                for i in range(len(grad_output[b][out_c])):
                    for j in range(len(grad_output[b][out_c][0])):
                        self.biases[out_c] -= learning_rate * grad_output[b][out_c][i][j]

        return grad_input


import numpy as np
class Conv2D_np:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernels = np.random.uniform(-0.1, 0.1, 
                          (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        self.biases = np.zeros(out_channels,dtype=np.float32)

    def forward(self, input):  # input: [batch, in_channels, height, width]
        self.input = input
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = np.pad(input, 
                           ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                           mode='constant').astype(np.float32)

        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w),dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            region = input[b, in_c, 
                                           i * self.stride:i * self.stride + k_h, 
                                           j * self.stride:j * self.stride + k_w]
                            output[b, out_c, i, j] += np.sum(region * self.kernels[out_c, in_c])
                output[b, out_c] += self.biases[out_c]

        return output

    def backward(self, grad_output, learning_rate):
        batch_size, _, out_h, out_w = grad_output.shape
        _, _, in_h, in_w = self.input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input_padded = np.pad(self.input, 
                                  ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                                  mode='constant').astype(np.float32)
        else:
            input_padded = self.input

        grad_input_padded = np.zeros_like(input_padded,dtype=np.float32)
        grad_kernels = np.zeros_like(self.kernels,dtype=np.float32)
        grad_biases = np.zeros_like(self.biases,dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            r = i * self.stride
                            c = j * self.stride
                            region = input_padded[b, in_c, r:r+k_h, c:c+k_w]
                            grad_kernels[out_c, in_c] += grad_output[b, out_c, i, j] * region
                            grad_input_padded[b, in_c, r:r+k_h, c:c+k_w] += self.kernels[out_c, in_c] * grad_output[b, out_c, i, j]
                grad_biases[out_c] += np.sum(grad_output[b, out_c])

        # Remove padding from grad_input if it was added
        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        # Update parameters
        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        return grad_input
