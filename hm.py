import sys
import numpy as np
import random
from time import time

dim = int(sys.argv[2])

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

# strassen
def split(matrix):
    row, col = matrix.shape
    row2, col2 = row // 2, col // 2
    return (matrix[:row2, :col2], matrix[:row2, col2:], 
    matrix[row2:, :col2], matrix[row2:, col2:])

def add_matrices(a, b):
    return [[x + y for x, y in zip(col_a, col_b)] for col_a, col_b in zip(a, b)]

def subtract_matrices(a, b):
    return [[x - y for x, y in zip(col_a, col_b)] for col_a, col_b in zip(a, b)]

def multiply_matrices(a, b):
    z_b = list(zip(*b))
    return [[sum(x * y for x, y in zip(a_row, b_col)) for b_col in z_b] for a_row in a]

def strassens(a, b, crossover=2):
    n = len(a)

    # base case
    if n <= crossover:
        return multiply_matrices(a, b)

    # recursive case
    else:
        range_n = range(n)
        # if not a power of 2
        if n%2 != 0:
            # find next power of 2
            new_n = n + 1

            # pad with zeroes
            a = [a[i] + [0] for i in range_n] + [[0] * new_n]
            b = [b[i] + [0] for i in range_n] + [[0] * new_n]

        else:
            new_n = n

        # where to split the matrix
        split = new_n//2
        first_half = range(0, split)
        second_half = range(split, new_n)

        # define sub-matrices
        A = [a[i][:split] for i in first_half]
        B = [a[i][split:new_n] for i in first_half]
        C = [a[i][:split] for i in second_half]
        D = [a[i][split:new_n] for i in second_half]
        E = [b[i][:split] for i in first_half]
        F = [b[i][split:new_n] for i in first_half]
        G = [b[i][:split] for i in second_half]
        H = [b[i][split:new_n] for i in second_half]

        # sub-multiplications
        P1 = strassens(A, subtract_matrices(F, H), crossover=crossover)
        P2 = strassens(add_matrices(A, B), H, crossover=crossover)
        P3 = strassens(add_matrices(C, D), E, crossover=crossover)
        P4 = strassens(D, subtract_matrices(G, E), crossover=crossover)
        P5 = strassens(add_matrices(A, D), add_matrices(E, H), crossover=crossover)
        P6 = strassens(subtract_matrices(B, D), add_matrices(G, H), crossover=crossover)
        P7 = strassens(subtract_matrices(A, C), add_matrices(E, F), crossover=crossover)

        # combine results
        result = list(map(lambda x,y:x+y, add_matrices(subtract_matrices(add_matrices(P5, P4), P2), P6), add_matrices(P1, P2)))
        result.extend(list(map(lambda x,y:x+y, add_matrices(P3, P4), subtract_matrices(subtract_matrices(add_matrices(P5, P1), P3), P7))))
        return [result[i][:n]for i in range_n]

# flag 1 is random, 0 is from input file
def test(dim):
    if sys.argv[1] == 1:
    # input text file with 2*dim^2 numbers representing matrices A, B
        with open(sys.argv[3]) as file:
            data = [int(line) for line in file]
        A = np.reshape(data[:dim**2], (dim, dim))
        B = np.reshape(data[dim**2:], (dim, dim))
    else:
        A, B= mat_gen(dim)
    #print(A, B)
    start_mult = time()
    c_mult = matrix_mult(A, B)
    end_mult = time()
    time_mult = end_mult-start_mult
    #print("mult", c_mult)
    print("Mult Time: ", time_mult)

    start_strass = time()
    c_strassen = strassens(A, B, crossover=30)
    end_strass = time()
    time_strass = end_strass-start_strass
    #print("strassen", c_strassen)
    print("Strass Time: ", time_strass)

test(dim)