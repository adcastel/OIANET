def matmul(A, B):
    # Matrix multiplication: A [m x n] * B [n x p] = C [m x p]
    m, n = len(A), len(A[0])
    p = len(B[0])
    C = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def transpose(M):
    return list(map(list, zip(*M)))


def vector_add(A, B):
    return [[a + b for a, b in zip(row, B)] for row in A]

def scalar_vector_add(A, B):
    return [a + b for a, b in zip(A, B)]
