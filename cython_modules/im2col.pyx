# im2col_cython.pyx
import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def im2col_forward_cython(np.ndarray[np.float32_t, ndim=4] input,
                   int kernel_size, int stride, int padding):
    cdef int B = input.shape[0]
    cdef int C = input.shape[1]
    cdef int H = input.shape[2]
    cdef int W = input.shape[3]
    cdef int k = kernel_size
    cdef int pad = padding
    cdef int s = stride

    cdef int padded_H = H + 2 * pad
    cdef int padded_W = W + 2 * pad
    cdef int out_h = (padded_H - k) // s + 1
    cdef int out_w = (padded_W - k) // s + 1

    cdef np.ndarray[np.float32_t, ndim=4] padded_input = np.zeros((B, C, padded_H, padded_W), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] cols = np.zeros((B, C * k * k, out_h * out_w), dtype=np.float32)

    cdef int b, c, i, j, m, n, out_idx, channel, patch_idx
    cdef int h_start, w_start

    # Apply padding
    for b in prange(B, nogil=True):
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    padded_input[b, c, i + pad, j + pad] = input[b, c, i, j]

    # Extract patches
    for b in range(B):
        out_idx = 0
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                w_start = j * s
                patch_idx = 0
                for c in range(C):
                    for m in range(k):
                        for n in range(k):
                            cols[b, patch_idx, out_idx] = padded_input[b, c, h_start + m, w_start + n]
                            patch_idx += 1
                out_idx += 1

    return cols
