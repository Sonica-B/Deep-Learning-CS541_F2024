import numpy as np


# def problem_1a (A, B):
#     return A + B
#
# def problem_1b (A, B, C):
#     return (np.dot(A,B) - C)
#
# def problem_1c (A, B, C):
#     return ((A * B) + np.transpose(C))
#
# def problem_1d (x, y):
#     return np.dot((np.transpose(x)),y)
#
# def problem_1e (A, i):
#     return np.sum(A[i,::2])

def problem_1f(A, c, d):
    return np.mean(A[np.nonzero((d >= A) & (A >=c))])


# def problem_1g (A, k):
#     return ...
#
# def problem_1h (x, k, m, s):
#     return ...
#
# def problem_1i (A):
#     return ...
#
# def problem_1j (x):
#     return ...
#
# def problem_1k (x, k):
#     return ...
#
# def problem_1l (X, Y):
#     return ...


A = np.array([[1, 4, 5, 12],
              [5, 8, 9, 0],
              [6, 7, 11, 19]])

print(problem_1f(A, 1, 10))
