import sys
import numpy as np
import random
from time import time



def matrix_mult(a, b):
    n = len(a)
    res = [[0 for y in range(n)] for x in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                res[i][j] += a[i][k] * b[k][j]
    return res

def mat_gen(dim):
    A = np.random.randint(-10,10, size=(dim, dim))
    B = np.random.randint(-10,10, size=(dim, dim))
    return A, B

p_set = [0.01, 0.02, 0.03, 0.04, 0.05]
# generate graphs
def rand_graph(p):
    V = 1024
    A = np.zeros((V, V))
    # A is adjacency matrix 

    # lol this is so brute force i'm crying sorry brain no work rn
    for i in range(V):
        for j in range(V):
            if random.random()< p:
                A[i][j] = 1
    return A

# strassen
def split(matrix):
    row, col = matrix.shape
    row2, col2 = row // 2, col // 2
    return (matrix[:row2, :col2], matrix[:row2, col2:], 
    matrix[row2:, :col2], matrix[row2:, col2:])

def strassen(x, y):
    if x.size == 1 or y.size == 1:
        return x * y

    n = x.shape[0]

    if n % 2 == 1:
        x = np.pad(x, (0, 1), mode='constant')
        y = np.pad(y, (0, 1), mode='constant')

    m = int(np.ceil(n / 2))
    a = x[: m, : m]
    b = x[: m, m:]
    c = x[m:, : m]
    d = x[m:, m:]
    e = y[: m, : m]
    f = y[: m, m:]
    g = y[m:, : m]
    h = y[m:, m:]
    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)
    result = np.zeros((2 * m, 2 * m), dtype=np.int32)
    result[: m, : m] = p5 + p4 - p2 + p6
    result[: m, m:] = p1 + p2
    result[m:, : m] = p3 + p4
    result[m:, m:] = p1 + p5 - p3 - p7

    return result[: n, : n]

# flag 1 is random, 0 is from input file
def test(dim):
    if sys.argv[1] == "1":
    # input text file with 2*dim^2 numbers representing matrices A, B
        with open(sys.argv[3]) as file:
            data = [int(line) for line in file]
        A = np.reshape(data[:dim**2], (dim, dim))
        B = np.reshape(data[dim**2:], (dim, dim))
    else:
        A, B= mat_gen(dim)

    print(A, B)
    start_mult = time()
    c_mult = matrix_mult(A, B)
    end_mult = time()
    time_mult = end_mult-start_mult
    #print("mult", c_mult)
    print("Mult Time: ", time_mult)

    start_strass = time()
    c_strassen = strassen(A, B)
    end_strass = time()
    time_strass = end_strass-start_strass
    #print("strassen", c_strassen)
    print("Strass Time: ", time_strass)

def triangle(p):
    A = rand_graph(p)
    triangles = strassen(strassen(A, A), A)
    return np.sum(triangles)

if sys.argv[1] != "2":
    print(sys.argv[1] == 2)
    dim = int(sys.argv[2])
    test(dim)
else:
    res = []
    for p in p_set:
        res.append(triangle(p))
    print(res)

