import numpy as np

def power_method(A, u0 = None, iter_end = 100):
    """
    Power Method

    :param A: a 2-dim NDArray
    :type A: np.ndarray
    :param u0: initial u for iteration, [1, 0, ..., 0] as default
    :type u0: array-like
    :param iter_end: the total iteration count
    :type iter_end: int
    :return: x which Ax = ax, a is the largest model eigenvalue
    :rtype: np.ndarray
    """
    n = A.shape[0]
    if u0 == None:
        u = np.zeros((n,1))
        u[0,0] = 1
    else:
        u = np.array(u0).reshape((n,1))
    for iter_count in range(iter_end):
        u = A@u
        u = u / max(abs(u))
    return u

def construct_companion_matrix(coefficients):
    """
    Transform the coefficients of Polynomial f(x) = x^n + a_{n-1}x^{n-1} + ... + a_0

    :param coefficients: a iterable table which `a[i] = a_i`
    :type coefficients: array-like
    :return: companion matrix of f(x)
    :rtype: np.ndarray
    """
    coefficients = np.array(coefficients)
    n = len(coefficients)
    companion_matrix = np.zeros((n, n))
    for i in range(1, n):
        companion_matrix[i, i - 1] = 1
    for i in range(n):
        companion_matrix[i, n - 1] = -coefficients[i]
    return companion_matrix

def p6_1():
    A = construct_companion_matrix((1,-5,3))
    u = power_method(A)
    print("Problem 6(2)(i):")
    print((A@u)[0,0]/u[0,0])
    A = construct_companion_matrix((0,-3,-1))
    u = power_method(A)
    print("Problem 6(2)(ii):")
    print((A@u)[0,0]/u[0,0])
    A = construct_companion_matrix((101,208.01,10891.01,9802.08,79108.9,-99902,790,-1000))
    u = power_method(A)
    print("Problem 6(2)(iii):")
    print((A@u)[0,0]/u[0,0])

def ez_test():
    print(construct_companion_matrix((1,1,3)))

if __name__ == "__main__":
    p6_1()