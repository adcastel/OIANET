from modules.layer import Layer

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):  # [batch][channels][h][w]
        self.input = input
        batch_size, channels = len(input), len(input[0])
        input_h, input_w = len(input[0][0]), len(input[0][0][0])

        out_h = (input_h - self.kernel_size) // self.stride + 1
        out_w = (input_w - self.kernel_size) // self.stride + 1

        output = []
        self.max_indices = []

        for b in range(batch_size):
            sample_out = []
            sample_indices = []
            for c in range(channels):
                channel_out = []
                channel_idx = []
                for i in range(out_h):
                    row = []
                    idx_row = []
                    for j in range(out_w):
                        window = []
                        coords = []
                        for ki in range(self.kernel_size):
                            for kj in range(self.kernel_size):
                                r = i * self.stride + ki
                                s = j * self.stride + kj
                                window.append(input[b][c][r][s])
                                coords.append((r, s))
                        max_val = max(window)
                        max_idx = window.index(max_val)
                        row.append(max_val)
                        idx_row.append(coords[max_idx])
                    channel_out.append(row)
                    channel_idx.append(idx_row)
                sample_out.append(channel_out)
                sample_indices.append(channel_idx)
            output.append(sample_out)
            self.max_indices.append(sample_indices)
        return output

    def backward(self, grad_output, learning_rate=None):
        batch_size = len(self.input)
        channels = len(self.input[0])
        input_h = len(self.input[0][0])
        input_w = len(self.input[0][0][0])

        # Output spatial dimensions after pooling
        pooled_h = len(grad_output[0][0])
        pooled_w = len(grad_output[0][0][0])

        # Initialize zero gradients for input shape
        grad_input = [[[[0 for _ in range(input_w)] for _ in range(input_h)]
                   for _ in range(channels)] for _ in range(batch_size)]

        for b in range(batch_size):
            for c in range(channels):
                for i in range(pooled_h):
                    for j in range(pooled_w):
                        r, s = self.max_indices[b][c][i][j]  # indices of max value during forward
                        grad_input[b][c][r][s] = grad_output[b][c][i][j]

        return grad_input

import numpy as np

class MaxPool2D_np(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):  # input: np.ndarray of shape [B, C, H, W]
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        output = np.zeros((B, C, out_h, out_w),dtype=input.dtype)

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW

                        window = input[b, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        max_val = window[max_idx]

                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

        return output

    def backward(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input