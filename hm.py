import sys
import numpy as np
from time import time

dim = int(sys.argv[2])

def mat_gen(dim):
    A = np.random.randint(-10,10, size=(dim, dim))
    B = np.random.randint(-10,10, size=(dim, dim))
    return A, B

def mat_mult(a, b):
    return [[sum(x * y for x, y in zip(a_r, b_c)) for b_c in zip(*b)] for a_r in a]

# strassen
def add(a, b):
    return [[x + y for x, y in zip(col_a, col_b)] for col_a, col_b in zip(a, b)]

def subtract(a, b):
    return [[x - y for x, y in zip(col_a, col_b)] for col_a, col_b in zip(a, b)]

def strassen(a, b, crossover=2):
    n = len(a)
    if n <= crossover:
        return mat_mult(a, b)
    else:
        range_n = range(n)
        if n%2 != 0:
            new_n = n + 1
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
        P1 = strassen(A, subtract(F, H), crossover=crossover)
        P2 = strassen(add(A, B), H, crossover=crossover)
        P3 = strassen(add(C, D), E, crossover=crossover)
        P4 = strassen(D, subtract(G, E), crossover=crossover)
        P5 = strassen(add(A, D), add(E, H), crossover=crossover)
        P6 = strassen(subtract(B, D), add(G, H), crossover=crossover)
        P7 = strassen(subtract(A, C), add(E, F), crossover=crossover)

        # combine results
        result = list(map(lambda x,y:x+y, add(subtract(add(P5, P4), P2), P6), add(P1, P2)))
        result.extend(list(map(lambda x,y:x+y, add(P3, P4), subtract(subtract(add(P5, P1), P3), P7))))
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
        A, B = mat_gen(dim)
    #print(A, B)
    start_mult = time()
    c_mult = mat_mult(A, B)
    end_mult = time()
    time_mult = end_mult-start_mult
    #print("mult", c_mult)
    print("Mult Time: ", time_mult)

    start_strass = time()
    c_strassen = strassen(A, B, crossover=30)
    end_strass = time()
    time_strass = end_strass-start_strass
    #print("strassen", c_strassen)
    print("Strass Time: ", time_strass)

test(dim)