import numpy as np
from . import p1_1
from math import *
import random


def square_root_method(A, b, advanced:bool=False, err=1e-10):
    """
    Return x from Ax=b

    :param A: Coefficient matrix
    :type A: np.ndarray
    :param b: Right-hand side vector
    :type b: np.ndarray
    :param advanced: Use advanced method
    :type advanced: bool
    :param err: Error tolerance
    :type err: float
    :return: Solution vector
    :rtype: np.ndarray
    """
    if A.shape[0] != A.shape[1] or A.ndim != 2:
        raise ValueError("Invalid A")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A.shape[0] != b.shape[0]")
    n=A.shape[0]
    x=np.zeros([n,1])
    # Calculate L
    if not advanced:
        L=np.zeros([n,n])
        for j in range(n):
            for i in range(j,n):
                if i == j:
                    L[i,j] = sqrt(abs(A[i,j] - sum(L[i,0:j]**2)))
                    # if abs(L[i,j]) < err:
                    #    raise ValueError("Invalid A, rankA != n")
                else:
                    L[i,j] = (A[i,j] - sum(L[j,0:j]*L[i,0:j])) / L[j,j]
        x=p1_1.solve_LUx_b(L, L.T, b)
    else:
        L=np.eye(n)
        D=np.zeros([n,n])
        for j in range(n):
            for i in range(j,n):
                if i == j:
                    D[i,j] = A[i,j] - L[i,0:j]@D[0:j,0:j]@L[j,0:j].T
                    # if abs(D[i,j]) < err:
                    #     raise ValueError("Invalid A, rankA != n")
                else:
                    L[i,j] = (A[i,j] - L[i,:]@D@L[j,:].T) / D[j,j]
        x=p1_1.solve_LUx_b(L@D, L.T, b)
    return x
    
def create_matrix_p1_2_1(a=0,c=10,b=None):
    """
    Random create b which every element is in [a,c], if you want determine b, give b as a parameter

    :param a: Lower bound
    :type a: float
    :param c: Upper bound
    :type c: float
    :param b: Predetermined vector
    :type b: np.ndarray
    :return: (A, b) where A is coefficient matrix and b is right-hand side vector
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if b == None:
        b_=np.zeros([100,1])
        for i in range(100):
            b_[i,0]=(c-a)*random.random()+a
    A=10*np.eye(100)
    for i in range(99):
        A[i,i+1]=1
        A[i+1,i]=1
    return A, b_

def create_matrix_p1_2_2(n=40):
    """
    Return A, b

    :param n: Matrix dimension
    :type n: int
    :return: (A, b) where A is coefficient matrix and b is right-hand side vector
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    b=np.zeros([n,1])
    A=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            A[i,j]=1/(i+1+j+1-1)
        b[i,0]=sum(A[i,:])
    return A, b

def p1_2_1(b=None):
    A, b=create_matrix_p1_2_1(b=b)
    x1=square_root_method(A,b)
    x2=square_root_method(A,b,advanced=True)
    print(x1.flatten())
    print(x2.flatten())
    p1_1.compare_ans({"Square root method":x1, "Advanced square root method":x2})

def p1_2_2(b=None):
    A, b=create_matrix_p1_2_2(40)
    x1=square_root_method(A,b)
    x2=square_root_method(A,b,advanced=True)
    print(x1.flatten())
    print(x2.flatten())
    p1_1.compare_ans({"Square root method":x1, "Advanced square root method":x2},False)

def p1_2():
    p1_2_1()
    p1_2_2()

def ez_test():
    A=np.array([[1,1,0],[1,2,0],[0,0,2]])
    b=np.array([[1], [2], [10]])
    print(square_root_method(A, b, True))

if __name__ == '__main__':
    p1_2()

