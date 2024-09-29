import numpy as np
from numpy import linalg as LA

def problem_1a (A, B):
    return (A + B)

def problem_1b (A, B, C):
    return (np.dot(A,B) - C)

def problem_1c (A, B, C):
    return (A*B) + np.transpose(C)

def problem_1d (x, y):
    return np.inner(x,y)

def problem_1e (A, i):
    return np.sum(A[i,::2])

def problem_1f (A, c, d):
    return np.mean(A[np.nonzero((A>=c) & (A<=d))])

def problem_1g (A, k):
    eigenvalues, eigenvectors = LA.eig(A)
    return eigenvectors[:,np.argsort(-np.abs(eigenvalues))[:k]]

def problem_1h (x, k, m, s):
    return np.random.multivariate_normal((x+m*np.ones((len(x)))).flatten(),(s*np.identity(len(x))),k).T

def problem_1i (A):
    return A[:, np.random.permutation(A.shape[1])]

def problem_1j (x):
    return (x - np.mean(x)) / np.std(x)

def problem_1k (x, k):
    return np.repeat(x[:, np.newaxis], k, axis=1)

def problem_1l (X, Y):
    return np.sqrt(np.sum((X[:, :, np.newaxis] - Y[:, np.newaxis, :]) ** 2, axis=0))


A = np.array([[1, 3], [5, 7]])
B = np.array([[4, 5], [6, 9]])
C = np.array([[2, 5], [4, 8]])
D = np.array([[1,3,4,12,15,34], [4, 2, 53,23,65, 83], [12,23,34,45,56,67]])

print("1a.",problem_1a(A,B))
print("\n 1b.",problem_1b(A,B,C))
print("\n 1c.",problem_1c(A,B,C))
print("\n 1d.",problem_1d(A,B))
print("\n 1e.",problem_1e(D,1))
print("\n 1f.",problem_1f(D,2,15))

arr = [[5,-10,-5],[2,14,2],[-4,-8,6]]
arr1 = np.diag((1, 2, 3))
print("\n 1g.",problem_1g(arr,1))
print("\n 1g.",problem_1g(arr1,2))

x = np.array([1,2,3,4,5]).T
k = 4
m = 2
s = 3

print("\n 1h.",problem_1h(x,4,2,3))
print("\n 1i.",problem_1i(D))
print("\n 1j.",problem_1j(D[0]))
print("\n 1k.",problem_1k(D[0],5))

X = np.array([[1, 3, 5], [2, 4, 6]])  # shape: 2x3
Y = np.array([[7, 9, 11, 13], [8, 10, 12, 14]])  # shape: 2x4
print("\n 1k.",problem_1l(X, Y)) # shape: 3x4