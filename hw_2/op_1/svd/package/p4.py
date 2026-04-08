import numpy as np
from . import p2_2
from typing import Literal
import time

def const_iteration_method_solve_Ax_b(
        A,
        b,
        method: Literal["Jacobi","Gauss-Seidel","SOR"] | None = None,
        iter_end = 1e-7,
        safety_check = True,
        **kwargs
        ):
    """
    Return x from the given Ax=b
    If you wish to customize paramter `M` and `g`, let the paramter `method` be None, and give `M` and `g` in `kwargs`
    If you wish to offer `w` in SOR iteration, let the paramter `method` be "SOR", and give `w` (or `w_opt`) in `kwargs`

    :param A: Coefficient matrix
    :type A: np.ndarray
    :param b: Right-hand side vector
    :type b: np.ndarray
    :param method: Iteration method: "Jacobi", "Gauss-Seidel", "SOR", or None
    :type method: str or None
    :param iter_end: Iteration convergence threshold
    :type iter_end: float
    :param safety_check: Check spectral radius convergence condition
    :type safety_check: bool
    :param kwargs: Additional parameters (M, g, w, w_opt, x, show_r, show_iter_info)
    :type kwargs: dict
    :return: Solution vector
    :rtype: np.ndarray
    """
    if method not in ["Jacobi","Gauss-Seidel","SOR",None]:
        raise ValueError("Method should be one of \"Jacobi\",\"Gauss-Seidel\",\"SOR\"")
    D = np.diag(np.diagonal(A))
    L = -np.tril(A, k=-1)
    U = -np.triu(A, k=1)
    inv_D = np.diag(1/np.diagonal(A))

    if method == "Jacobi":
        M = inv_D@(L+U)
        g = inv_D@b

    elif method == "Gauss-Seidel":
        inv_DmL =  np.linalg.inv(D-L)
        M = inv_DmL@U
        g = inv_DmL@b

    elif method == 'SOR':
        if "w" in kwargs:
            w = kwargs["w"]
        elif "w_opt" in kwargs:
            w = kwargs["w_opt"]
        else :
            w = estimate_optimized_w(A)
        M = np.linalg.inv(D-w*L)@((1-w)*D+w*U)
        g = w*np.linalg.inv(D-w*L)@b

    else :
        M = kwargs['M']
        g = kwargs['g']

    n = M.shape[0]
    if 'x' in kwargs:
        x = kwargs['x']
    else:
        x = np.zeros((n,1))
    
    if safety_check :
        q = spectral_radius(M)
        if q >= 1:
            raise ValueError("M formed by A is not so good to converge.(\\rho(M)=%f)"%(q))
        
    # main iteration
    x_1 = M@x+g
    iter_count = 1
    r=[]
    start_time = time.time()
    while np.linalg.norm(x-x_1,2) >= iter_end:
        x = x_1
        x_1 = M@x+g
        iter_count += 1
        r.append(np.linalg.norm(A@x_1-b))

    if 'show_r' in kwargs:
        if kwargs["show_r"]:
            p2_2.show_graph({"r":r})
    if 'show_iter_info' in kwargs:
        if kwargs["show_iter_info"]:
            print("Iterated %d times."%(iter_count))
            end_time = time.time()
            execution_time = end_time - start_time
            print("Iterated %f seconds."%(execution_time))

    return x_1

def estimate_optimized_w(A, n = 2000, a=0, b=2 , show_graph = False):
    """
    Estimate the w where the spectral radius of L_w (given by A in SOR iteration) is minimum

    :param A: matrix A in SOR iteration
    :type A: np.ndarray
    :param n: the total hits that the value w will try in [a,b)
    :type n: int
    :param a: Lower bound of search range
    :type a: float
    :param b: Upper bound of search range
    :type b: float
    :param show_graph: Display optimization graph
    :type show_graph: bool
    :return: Optimal relaxation parameter
    :rtype: float
    """
    min_rho = 1
    w_opt = 1
    D = np.diag(np.diagonal(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    w_list = []
    rho_list = []
    for i in range(n):
        w=i/n*(b-a)+a
        L_w=np.linalg.inv(D-w*L)@((1-w)*D+w*U)
        rho=spectral_radius(L_w)

        w_list.append(w)
        rho_list.append(rho)

        if rho < min_rho:
            min_rho = rho
            w_opt = w
    print("w_opt = %f, \\rho(L_w) = %f"%(w_opt, min_rho))
    if show_graph:
        p2_2.show_graph({"x":w_list,r"$\rho(L_w)$":rho_list},False)
    return w_opt


def spectral_radius(matrix):
    """
    Get the \\rho(matrix)

    :param matrix: Input matrix
    :type matrix: np.ndarray
    :return: Spectral radius
    :rtype: float
    """
    eigenvalues = np.linalg.eigvals(matrix)
    radius = max(np.abs(eigenvalues))
    return radius

def tridiagonal_matrix(n, a, b, c):
    """
    Form a n-order tridiagonal matrix like below:
    [b, c, 0, ..., 0]
    [a, b, c, ..., 0]
    [0, a, b, ..., 0]
    [..., ..., ..., ..., ...]
    [0, 0, 0, ..., b]

    :param n: Matrix dimension
    :type n: int
    :param a: Lower diagonal element
    :type a: float
    :param b: Main diagonal element
    :type b: float
    :param c: Upper diagonal element
    :type c: float
    :return: Tridiagonal matrix
    :rtype: np.ndarray
    """
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, b)
    for i in range(1, n):
        matrix[i, i-1] = a
    for i in range(n-1):
        matrix[i, i+1] = c
    return matrix

def p4_1():
    a=0.5
    n=100
    h=1/n
    for (e,w) in [(1,1.939500),(0.1,1.892800),(0.01,1.497000),(0.0001,1.009400)]:
        A=tridiagonal_matrix(n-1, e, -(2*e+h), e+h)
        b=np.ones([n-1,1])*a*h*h
        b[n-2,0]=a*h*h - (e+h)*1
        hit_list = [x/n for x in range(1,n)]
        x_0 = np.round(np.array([(1-a)*(1-np.exp(-x/e))/(1-np.exp(-1/e))+a*x for x in hit_list]),4)
        x_1 = np.round(const_iteration_method_solve_Ax_b(A,b,'Jacobi').flatten(),4)
        x_2 = np.round(const_iteration_method_solve_Ax_b(A,b,"Gauss-Seidel").flatten(),4)
        x_3 = np.round(const_iteration_method_solve_Ax_b(A,b,"SOR",w = w).flatten(),4)
        p2_2.show_graph(
            {
                "x":hit_list,
                "Expected":x_0,
                "Jacobi":x_1,
                "G-S":x_2,
                "SOR":x_3
            },
            title="$\epsilon = %f$"%(e)
        )
        p2_2.show_graph(
            {
                "x":hit_list,
                "Jacobi-Expected":x_1-x_0,
                "G-S-Expected":x_2-x_0,
                "SOR-Expected":x_3-x_0
            },
            title="$\epsilon = %f$"%(e)
        )


def p4_2():
    # n = $N$
    for n in [20,40,80]:
        print("Test Branch: N = %d:"%(n))
        h = 1/n
        A = 4*np.eye((n-1)*(n-1))
        b = np.zeros(((n-1)*(n-1),1))
        Np = np.zeros((n-1,n-1))
        Nn = np.zeros((n-1,n-1))
        v=np.ones((n-1,1))
        s=np.zeros((n-1,1))
        s[0,0]=1
        s[n-2,0]=1
        for i in range(n-2):
            Np[i,i+1] = 1
            Nn[i+1,i] = 1
            A[i*(n-1):(i+1)*(n-1),(i+1)*(n-1):(i+2)*(n-1)] = -np.eye(n-1)
            A[(i+1)*(n-1):(i+2)*(n-1),i*(n-1):(i+1)*(n-1)] = np.eye(n-1)
        G = []
        F = []
        for i in range(n-1):
            G.append(np.exp(i*h)*np.diag([np.exp(j*h) for j in range(n-1)]))
            F.append(i*h*np.array([[j*h for j in range(n-1)]]).T)
        for i in range(n-1):
            A[i*(n-1):(i+1)*(n-1),i*(n-1):(i+1)*(n-1)]+=h*h*G[i]-Np-Nn
            b[i*(n-1):(i+1)*(n-1),:]=h*h*F[i]+s
        b[0:n-1,:]-=v
        b[(n-2)*(n-1):(n-1)*(n-1),:]+=v
        
        const_iteration_method_solve_Ax_b(A,b,'Gauss-Seidel',safety_check=False,show_iter_info = True)
        
def p4():
    p4_1()
    p4_2()

def find_w():
    n=100
    e=0.0001
    h=1/n
    A=tridiagonal_matrix(n+1, e, -(2*e+h), e+h)
    estimate_optimized_w(A,1000,1,1.01,True)

if __name__ == '__main__':
    p4()