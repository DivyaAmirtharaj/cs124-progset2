# read in inputs

import sys
import numpy as np
import random

with open(sys.argv[3]) as file:
    data = [int(line) for line in file]

dim = sys.argv[2]
A = np.reshape(data[:dim**2], (dim, dim))
B = np.reshape(data[dim**2:], (dim, dim))
print(A, B)

def split(matrix):
    row, col = matrix.shape
    row2, col2 = row // 2, col // 2
    return (matrix[:row2, :col2], matrix[:row2, col2:], 
    matrix[row2:, :col2], matrix[row2:, col2:])

def strassen(x, y):
    if len(x) == 1:
        return x * y
    a, b, c, d = split(x)
    e, f, g, h = split(y)


    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)

    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7

    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))

    return c

def naive(x, y):
    res = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            res[i][j] = 0
            for k in range(dim):
                res[i][j] += x[i][k] * B[k][j]
    return res

# generate random matrices
def rand(dim):
    return np.random.randint(-1, 1, (dim, dim))
    # or (0, 2, (dim, dim))

p_set = [0.01, 0.02, 0.03, 0.04, 0.05]
# generate graphs
def rand_graph(p):
    V = 1024
    A = np.zeros((V, V))
    # A is adjacency matrix 

    # lol this is so brute force i'm crying sorry brain no work rn
    for i in V:
        for j in V:
            if random.random()< p:
                A[i][j] = 1
    return A

