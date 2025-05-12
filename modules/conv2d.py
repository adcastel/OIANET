from modules.layer import Layer
from modules.utils import *
from cython_modules.im2col import im2col_forward_cython

import numpy as np

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_algo=0, weight_init="he"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if conv_algo == 0:
            self.mode = 'direct' 
        elif conv_algo == 1: 
            self.mode = 'im2col' # 'direct' or 'im2col'
        else:
            self.mode = 'fused' # 'direct' or 'im2col'
        
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size

        if weight_init == "he":
            std = np.sqrt(2.0 / fan_in)
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "custom":
            self.kernels = np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
        else:
            self.kernels = np.random.uniform(-0.1, 0.1, 
                          (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        

        self.biases = np.zeros(out_channels, dtype=np.float32)

        self.mc = 480
        self.nc = 3072
        self.kc = 384
        self.mr = 32
        self.nr = 12
        self.Ac = np.empty((self.mc, self.kc), dtype=np.float32)
        self.Bc = np.empty((self.kc, self.nc), dtype=np.float32)


    def get_weights(self):
        return {'kernels': self.kernels, 'biases': self.biases}

    def set_weights(self, weights):
        self.kernels = weights['kernels']
        self.biases = weights['biases']
    
    def forward(self, input, training=True):
        self.input = input
        if self.mode == 'direct':
            return self._forward_direct(input)
        elif self.mode == 'im2col':
            return self._forward_im2col(input)
        elif self.mode == 'fused':
            return self._forward_im2col_fused(input)
        else:
            raise ValueError("Mode must be 'direct' or 'im2col'")

    def backward(self, grad_output, learning_rate):
        if self.mode == 'direct':
            return self._backward_direct(grad_output, learning_rate)
        elif self.mode == 'im2col':
            return self._backward_im2col(grad_output, learning_rate)
        elif self.mode == 'fused':
            return self._backward_im2col_fused(grad_output, learning_rate)
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
        return im2col_forward_cython(input, self.kernel_size, self.stride, self.padding)
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
    
    # --- IM2COL WITH GEMM IMPLEMENTATION ---

    def _col2im(self, cols, input_shape):
        batch_size, in_c, in_h, in_w = input_shape
        k = self.kernel_size
        out_h = (in_h - k + 2 * self.padding) // self.stride + 1
        out_w = (in_w - k + 2 * self.padding) // self.stride + 1

        padded_h = in_h + 2 * self.padding
        padded_w = in_w + 2 * self.padding
        padded_input = np.zeros((batch_size, in_c, padded_h, padded_w), dtype=np.float32)

        for b in range(batch_size):
            col = 0
            for i in range(out_h):
                for j in range(out_w):
                    patch = cols[b, :, col].reshape(in_c, k, k)
                    padded_input[b, :, 
                             i * self.stride:i * self.stride + k, 
                             j * self.stride:j * self.stride + k] += patch
                    col += 1

        if self.padding > 0:
            return padded_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return padded_input

    
    
    def _forward_im2col_fused(self, input):
        self.input = input
        self.cols = self._im2col(input)  # [B, Ck², OH*OW]
        B, Ck2, HW = self.cols.shape
        OC = self.out_channels
        kernel_matrix = self.kernels.reshape(OC, Ck2)

        # Flatten batch into columns: B * HW
        fused_HW = B * HW
        fused_cols = self.cols.transpose(1, 0, 2).reshape(Ck2, fused_HW)
        # Output will be [OC, B * HW]
        output = np.zeros((OC, fused_HW), dtype=np.float32)

        # GEMM 3-loop: output[oc, idx] = dot(kernel[oc], col[:, idx])
        """
        ORI
        for i in range(OC):
            for j in range(fused_HW):
                sum_val = 0.0
                for k in range(Ck2):
                    sum_val += kernel_matrix[i][k] * fused_cols[k][j]
                output[i][j] = sum_val + self.biases[i]
        
        """
        
        #output = matmul_goto(self, fused_HW, Ck2, OC, fused_cols, kernel_matrix, output)
        #for i in range(OC):
        #    for j in range(fused_HW):
        #        output[i][j] = output[i][j] + self.biases[i]
        
        
        
        # Mixed macro-kernel goto and micro-kernel Numpy
        #output = matmul_goto_np(self, fused_HW, Ck2, OC, fused_cols, kernel_matrix, output)
        #for i in range(OC):
        #    for j in range(fused_HW):
        #        output[i][j] = output[i][j] + self.biases[i]

        # Numpy GEMM 
        
        output += kernel_matrix @ fused_cols + self.biases.reshape(OC, 1)  # [OC, B*HW]
   
        #for i in range(OC):
        #    for j in range(fused_HW):
        #        output[i][j] = output[i][j] + self.biases[i]

        # Reshape back: [OC, B, HW] → [B, OC, OH, OW]
        output = output.reshape(OC, B, HW).transpose(1, 0, 2)
        out_h = (input.shape[2] - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (input.shape[3] - self.kernel_size + 2 * self.padding) // self.stride + 1
        return output.reshape(B, OC, out_h, out_w)

    def _backward_im2col_fused(self, grad_output, learning_rate):
        B, OC, OH, OW = grad_output.shape
        grad_output = grad_output.reshape(B, OC, OH * OW)  # [B, OC, HW]
        grad_output_fused = grad_output.transpose(1, 0, 2).reshape(OC, B * OH * OW)  # [OC, B*HW]

        Ck2 = self.kernels.shape[1] * self.kernels.shape[2] * self.kernels.shape[3]  # C * K * K
        IC = self.kernels.shape[1]

        kernel_matrix = self.kernels.reshape(OC, Ck2)
        grad_kernels = np.zeros_like(kernel_matrix, dtype=np.float32)
        grad_cols = np.zeros((Ck2, B * OH * OW), dtype=np.float32)

        cols_fused = self.cols.transpose(1, 0, 2).reshape(Ck2, B * OH * OW)  # [Ck², B*HW]

        # Compute grad_kernels and grad_cols using 3-loop GEMM-style
        #for i in range(OC):
        #    for j in range(B * OH * OW):
        #        for k in range(Ck2):
        #            grad_kernels[i][k] += grad_output_fused[i][j] * cols_fused[k][j]
        #            grad_cols[k][j] += kernel_matrix[i][k] * grad_output_fused[i][j]
        grad_kernels += grad_output_fused @ cols_fused.T
        grad_cols += kernel_matrix.T @ grad_output_fused

        # Compute grad_input from grad_cols
        grad_cols_batched = grad_cols.reshape(Ck2, B, OH * OW).transpose(1, 0, 2)  # [B, Ck², OH*OW]
        grad_input = self._col2im(grad_cols_batched, self.input.shape)

        # Bias gradient: sum over all gradients per output channel
        grad_biases = np.sum(grad_output_fused, axis=1)  # [OC]

        # Update parameters
        self.kernels -= learning_rate * grad_kernels.reshape(self.kernels.shape)
        self.biases -= learning_rate * grad_biases

        return grad_input
