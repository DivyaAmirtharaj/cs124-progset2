import sys
import numpy as np
import random
from time import time
from math import ceil, floor, log

# flag: arg 1, 0 = random matrices w fixed crossover, 1 = input file w fixed crossover
# 2 = triangles, 3 = experimental crossover

dim = int(sys.argv[2])
if dim % 2 == 0:
    crossover = 12
else:
    crossover = 37

# generating random matrices
def mat_gen(dim):
    mat1 = np.random.randint(-10,10, size=(dim, dim))
    mat2 = np.random.randint(-10,10, size=(dim, dim))
    return mat1, mat2

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

# regular matrix multiplicatoin
def mat_mult(a, b):
    return [[sum(x * y for x, y in zip(a_x, b_y)) for b_y in zip(*b)] for a_x in a]

# strassen
def add(mat1, mat2):
    return [[x + y for x, y in zip(col_mat1, col_mat2)] for col_mat1, col_mat2 in zip(mat1, mat2)]

def subtract(mat1, mat2):
    return [[x - y for x, y in zip(col_mat1, col_mat2)] for col_mat1, col_mat2 in zip(mat1, mat2)]

def strassen(mat1, mat2, crossover):
    n = len(mat1)
    if n <= crossover:
        return mat_mult(mat1, mat2)
    else:
        range_n = range(n)
        # pad for odd matrices
        if floor(log(n, 2)) != ceil(log(n, 2)):
            pow2 = int(ceil(log(n, 2)))
            n = 2**pow2
            #print(n)
            temp1 = [[0 for i in range(n)] for j in range(n)]
            temp2 = [[0 for i in range(n)] for j in range(n)]
            for i in range(len(mat1)):
                for j in range(len(mat1)):
                    temp1[i][j] = mat1[i][j]
                    temp2[i][j] = mat2[i][j]
            mat1 = temp1
            mat2 = temp2

        # submatrices
        split = n // 2
        front = range(0, split)
        back = range(split, n)

        # define sub-matrices
        A = [mat1[i][ :split] for i in front]
        B = [mat1[i][split:n] for i in front]
        C = [mat1[i][ :split] for i in back]
        D = [mat1[i][split:n] for i in back]
        E = [mat2[i][ :split] for i in front]
        F = [mat2[i][split:n] for i in front]
        G = [mat2[i][ :split] for i in back]
        H = [mat2[i][split:n] for i in back]

        # small 
        m1 = strassen(A, subtract(F, H), crossover=crossover)
        m2 = strassen(add(A, B), H, crossover=crossover)
        m3 = strassen(add(C, D), E, crossover=crossover)
        m4 = strassen(D, subtract(G, E), crossover=crossover)
        m5 = strassen(add(A, D), add(E, H), crossover=crossover)
        m6 = strassen(subtract(B, D), add(G, H), crossover=crossover)
        m7 = strassen(subtract(A, C), add(E, F), crossover=crossover)

        # combine results
        result = list(map(lambda x,y:x+y, add(subtract(add(m5, m4), m2), m6), add(m1, m2)))
        result.extend(list(map(lambda x,y:x+y, add(m3, m4), subtract(subtract(add(m5, m1), m3), m7))))
        # print(np.shape([result[i][:n]for i in range_n]))
        return [result[i][:n]for i in range_n]

# triangle
def triangle(p):
    total = 0
    for i in range(5):
        A = rand_graph(p)
        triangles = strassen(strassen(A, A, crossover), A, crossover)
        total += np.trace(triangles)/6
    return total / 5

# timer function
def timer(func, *args):
    start = time()
    run = func(*args)
    end = time()
    res = end - start 
    return res

# flag 0 is from input file, 1 is random: 3 is experimental crossover, 2 is 
def strass_run(dim):
    if sys.argv[1] == "0":
        # print("why won't you work")
        # input text file with 2*dim^2 numbers representing matrices A, B
        with open(sys.argv[3]) as file:
            data = [int(line) for line in file]
        A = np.reshape(data[:dim**2], (dim, dim))
        B = np.reshape(data[dim**2:], (dim, dim))
    else:
        A, B = mat_gen(dim)
    #print(A, B)
    res = strassen(A, B, crossover)
    #correct = mat_mult(A, B)
    # print("Mult Time: ", timer(mat_mult, A, B))
    # print("Strass Time: ", timer(strassen, A, B, crossover))
    for i in range(len(A)):
        #print(res[i][i], correct[i][i])
        print(res[i][i])

def test():
    if sys.argv[1] != "2":
        strass_run(dim)
        #print("hello")
    else:
        res = []
        for p in p_set:
            res.append(triangle(p))
        print(res)

# find the experimental crossover
def experiment():
    A, B = mat_gen(dim)
    print("Mult Time: ", timer(mat_mult, A, B))
    for i in range(10, 40):
        print("Strass Time: ", timer(strassen, A, B, i))

if sys.argv[1] == "3":
    experiment()
else:
    test()