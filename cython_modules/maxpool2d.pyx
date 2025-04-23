from cython.parallel import prange
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def maxpool_forward_cython(np.ndarray[np.float32_t, ndim=4] input,
                    int kernel_size, int stride):
    cdef int B = input.shape[0]
    cdef int C = input.shape[1]
    cdef int H = input.shape[2]
    cdef int W = input.shape[3]
    cdef int KH = kernel_size
    cdef int KW = kernel_size
    cdef int SH = stride
    cdef int SW = stride

    cdef int out_h = (H - KH) // SH + 1
    cdef int out_w = (W - KW) // SW + 1

    cdef np.ndarray[np.float32_t, ndim=4] output = np.zeros((B, C, out_h, out_w), dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=5] max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=np.int32)

    cdef int b, c, i, j, m, n, h_start, w_start, h_end, w_end
    cdef int max_h_idx, max_w_idx
    cdef float max_val, val

    for b in prange(B, nogil=True):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * SH
                    w_start = j * SW
                    h_end = h_start + KH
                    w_end = w_start + KW

                    max_val = -1e10  # assume input is float32
                    max_h_idx = 0
                    max_w_idx = 0

                    for m in range(h_start, h_end):
                        for n in range(w_start, w_end):
                            val = input[b, c, m, n]
                            if val > max_val:
                                max_val = val
                                max_h_idx = m
                                max_w_idx = n

                    output[b, c, i, j] = max_val
                    max_indices[b, c, i, j, 0] = max_h_idx
                    max_indices[b, c, i, j, 1] = max_w_idx

    return output, max_indices
