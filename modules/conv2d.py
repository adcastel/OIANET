from modules.layer import Layer
import random

import numpy as np

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_im2col=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode = 'direct' if use_im2col == False else 'im2col' # 'direct' or 'im2col'

        self.kernels = np.random.uniform(-0.1, 0.1, 
                          (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        self.biases = np.zeros(out_channels, dtype=np.float32)

    def forward(self, input):
        self.input = input
        if self.mode == 'direct':
            return self._forward_direct(input)
        elif self.mode == 'im2col':
            return self._forward_im2col(input)
        else:
            raise ValueError("Mode must be 'direct' or 'im2col'")

    def backward(self, grad_output, learning_rate):
        if self.mode == 'direct':
            return self._backward_direct(grad_output, learning_rate)
        elif self.mode == 'im2col':
            return self._backward_im2col(grad_output, learning_rate)
        else:
            raise ValueError("Mode must be 'direct' or 'im2col'")

    # --- DIRECT IMPLEMENTATION ---

    def _forward_direct(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = np.pad(input,
                           ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                           mode='constant').astype(np.float32)

        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)

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

    def _backward_direct(self, grad_output, learning_rate):
        batch_size, _, out_h, out_w = grad_output.shape
        _, _, in_h, in_w = self.input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input_padded = np.pad(self.input,
                                  ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                  mode='constant').astype(np.float32)
        else:
            input_padded = self.input

        grad_input_padded = np.zeros_like(input_padded, dtype=np.float32)
        grad_kernels = np.zeros_like(self.kernels, dtype=np.float32)
        grad_biases = np.zeros_like(self.biases, dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            r = i * self.stride
                            c = j * self.stride
                            region = input_padded[b, in_c, r:r + k_h, c:c + k_w]
                            grad_kernels[out_c, in_c] += grad_output[b, out_c, i, j] * region
                            grad_input_padded[b, in_c, r:r + k_h, c:c + k_w] += self.kernels[out_c, in_c] * grad_output[b, out_c, i, j]
                grad_biases[out_c] += np.sum(grad_output[b, out_c])

        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        return grad_input

    # --- IM2COL IMPLEMENTATION ---

    def _im2col(self, input):
        batch_size, in_c, in_h, in_w = input.shape
        k = self.kernel_size
        out_h = (in_h - k + 2 * self.padding) // self.stride + 1
        out_w = (in_w - k + 2 * self.padding) // self.stride + 1

        if self.padding > 0:
            input = np.pad(input,
                           ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                           mode='constant')

        cols = np.zeros((batch_size, in_c * k * k, out_h * out_w), dtype=np.float32)
        for b in range(batch_size):
            col = 0
            for i in range(out_h):
                for j in range(out_w):
                    patch = input[b, :, i * self.stride:i * self.stride + k, j * self.stride:j * self.stride + k]
                    cols[b, :, col] = patch.reshape(-1)
                    col += 1
        return cols

    def _forward_im2col(self, input):
        self.input = input
        self.cols = self._im2col(input)  # [B, Ck², OH*OW]
        batch_size = input.shape[0]
        out_h = (input.shape[2] - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (input.shape[3] - self.kernel_size + 2 * self.padding) // self.stride + 1
        kernel_matrix = self.kernels.reshape(self.out_channels, -1)  # [OC, Ck²]

        output = np.zeros((batch_size, self.out_channels, out_h * out_w), dtype=np.float32)

        # GEMM via 3-loop
        for b in range(batch_size):
            for i in range(self.out_channels):
                for j in range(out_h * out_w):
                    sum_val = 0.0
                    for k in range(kernel_matrix.shape[1]):
                        sum_val += kernel_matrix[i][k] * self.cols[b][k][j]
                    output[b][i][j] = sum_val + self.biases[i]

        return output.reshape(batch_size, self.out_channels, out_h, out_w)

    def _backward_im2col(self, grad_output, learning_rate):
        batch_size, out_channels, out_h, out_w = grad_output.shape
        grad_output_reshaped = grad_output.reshape(batch_size, out_channels, -1)
        grad_kernels = np.zeros_like(self.kernels, dtype=np.float32)
        grad_biases = np.zeros_like(self.biases, dtype=np.float32)
        grad_cols = np.zeros_like(self.cols, dtype=np.float32)
        kernel_matrix = self.kernels.reshape(self.out_channels, -1)

        for b in range(batch_size):
            for i in range(self.out_channels):
                for j in range(out_h * out_w):
                    grad_biases[i] += grad_output_reshaped[b][i][j]
                    for k in range(kernel_matrix.shape[1]):
                        grad_kernels[i][k // (self.kernel_size ** 2)][(k % (self.kernel_size ** 2)) // self.kernel_size][(k % self.kernel_size)] += self.cols[b][k][j] * grad_output_reshaped[b][i][j]
                        grad_cols[b][k][j] += kernel_matrix[i][k] * grad_output_reshaped[b][i][j]

        # Update weights
        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        # col2im
        grad_input = np.zeros_like(self.input, dtype=np.float32)
        if self.padding > 0:
            grad_input_padded = np.zeros((batch_size, self.in_channels,
                                          self.input.shape[2] + 2 * self.padding,
                                          self.input.shape[3] + 2 * self.padding), dtype=np.float32)
        else:
            grad_input_padded = grad_input

        for b in range(batch_size):
            col = 0
            for i in range(out_h):
                for j in range(out_w):
                    patch = grad_cols[b, :, col].reshape(self.in_channels, self.kernel_size, self.kernel_size)
                    grad_input_padded[b, :, i * self.stride:i * self.stride + self.kernel_size,
                                         j * self.stride:j * self.stride + self.kernel_size] += patch
                    col += 1

        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return grad_input
