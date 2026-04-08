import numpy as np
from . import p1_1

def climbing_blind_estimation(A, x0=None):
    """
    Estimate the 1-norm of A

    :param A: a numpy matrix, ndim==2, square
    :type A: np.ndarray
    :param x0: (optional) give a initial point to the iteration, (1/n, ..., 1/n)^T by default, n==A.shape[0]
    :type x0: np.ndarray
    :return: Estimated 1-norm of A
    :rtype: float
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A should be a square!")
    n = A.shape[0]
    if x0 == None:
        x0 = np.ones([n,1])/n
    x = x0
    while True:
        w = A@x
        v = np.sign(w)
        z = (A.T)@v
        if max(abs(z)) <= (z.T)@x:
            return np.sum(abs(w))
        else:
            j = np.argmax(abs(z))
            x = np.zeros([n,1])
            x[j, 0] = 1

def climbing_blind_estimation_inverse(A, x0=None):
    """
    Estimate the infinite-norm of the inverse of A

    :param A: a numpy matrix, ndim==2, square
    :type A: np.ndarray
    :param x0: (optional) give a initial point to the iteration, (1/n, ..., 1/n)^T by default, n==A.shape[0]
    :type x0: np.ndarray
    :return: the estimated infinite-norm of the inverse of A
    :rtype: float
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A should be a square!")
    n = A.shape[0]
    if x0 == None:
        x0 = np.ones([n,1])/n
    x = x0
    L, U, P = p1_1.gauss_elimation(A, 'column')
    Lt, Ut, Pt = p1_1.gauss_elimation(A.T, 'column')
    while True:
        # w = B@x <==> A.T@z = v
        w = p1_1.solve_LUx_b(Lt, Ut, Pt@x)
        v = np.sign(w)
        # z = (B.T)@v <==> A@z = v
        z = p1_1.solve_LUx_b(L, U, P@v)
        if max(abs(z)) <= (z.T)@x:
            return np.sum(abs(w))
        else:
            j = np.argmax(abs(z))
            x = np.zeros([n,1])
            x[j, 0] = 1

def create_hilbert_matrix(n=40):
    """
    Return H

    :param n: Matrix dimension
    :type n: int
    :return: Hilbert matrix
    :rtype: np.ndarray
    """
    A=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            A[i,j]=1/(i+1+j+1-1)
    return A

def p2_1():
    for n in range(5,21):
        A=create_hilbert_matrix(n)
        k=climbing_blind_estimation_inverse(A)*np.linalg.norm(A,ord=np.inf)
        print("Where order is %2d, k = %e"%(n, k))

    
if __name__ == "__main__":
    """
    A=np.eye(3)
    A[2,1]=-1/6
    print(A)
    print(climbing_blind_estimation_inverse(A))
    """
    p2_1()
