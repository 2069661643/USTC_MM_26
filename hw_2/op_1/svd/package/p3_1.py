from . import p1_1, p1_2, p2_2
import numpy as np

def householder_transformation(x):
    """
    Return v, b, s.t. H@x = re_1, H = I-bv.T@v

    :param x: Input vector
    :type x: np.ndarray
    :return: (v, b) where v is Householder vector and b is Householder coefficient
    :rtype: tuple[np.ndarray, float]
    """
    x = x.copy()
    n = x.shape[0]
    nu = np.linalg.norm(x,np.inf)
    if nu == 0:
        # raise ValueError("nu=0")
        return x, 0.0
    x = x/nu
    sigma = x[1:,0].T @ x[1:,0]
    v = x.copy()
    if sigma == 0 :
        b = 0
    else:
        alpha = np.sqrt(x[0,0]**2 + sigma)
        if x[0,0] <= 0 :
            v[0,0] = x[0,0] - alpha
        else :
            v[0,0] = -sigma/(x[0,0]+alpha)
        b = 2*v[0,0]*v[0,0]/(sigma+v[0,0]*v[0,0])
        v = v/v[0,0]
    return v,b

def house(x):
    """
    Return v, b, s.t. H@x = re_1, H = I-bv.T@v

    :param x: Input vector
    :type x: np.ndarray
    :return: (v, b) where v is Householder vector and b is Householder coefficient
    :rtype: tuple[np.ndarray, float]
    """
    return householder_transformation(x)

def QR_decomposition(A):
    """
    Return A, d, which component can be found on P95

    :param A: Input matrix
    :type A: np.ndarray
    :return: (A, d) where A is modified matrix containing Q and R, d is Householder coefficients
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    A=A.copy()
    m, n = A.shape
    d = np.zeros((n,1))
    for j in range(n):
        if j == m-1 :
            break
        if j == n-2:
            1
        v, b = householder_transformation(A[j:m,j:j+1])
        A[j:m,j:n]=(np.eye(m-j)-b*v@v.T)@A[j:m,j:n]
        d[j,0]=b
        # print(A[j+1:m,j])
        # print(v[1:m-j+1,0])
        A[j+1:m,j]=v[1:m-j+1,0]
        # print(A[j+1:m,j])
        # print(v[1:m-j+1,0])
    return A, d

def QR_solve_Ax_b(A,b,QR_decomposition_tuple=None):
    """
    Solve Ax=b with QR decomposition
    Remark: Offering a parameter "QR_decomposition_tuple" can skip the calculation of the function QR_decomposition()

    :param A: Coefficient matrix
    :type A: np.ndarray
    :param b: Right-hand side vector
    :type b: np.ndarray
    :param QR_decomposition_tuple: Precomputed QR decomposition (A, d)
    :type QR_decomposition_tuple: tuple
    :return: Solution vector
    :rtype: np.ndarray
    """
    if QR_decomposition_tuple != None:
        A,d = QR_decomposition_tuple 
    else :
        A,d = QR_decomposition(A)
    m, n = A.shape
    for j in range(n):
        if j == m-1 :
            break
        v = np.zeros((m,1))
        v[j,0]=1
        v[j+1:,0]=A[j+1:,j]
        beta=d[j,0]
        b = (np.eye(m)-beta*v@v.T)@b
    if A[n-1,n-1] < 1e-10:
        A[n-1,n-1] = 1e-10
    return p1_1.solve_LUx_b(np.eye(m),A,b)

def Solution_of_LS_problem(A,b,QR_decomposition_tuple=None):
    """
    Get argmin_x(np.linalg.norm(A@x-b,2))
    Remark: Offering a parameter "QR_decomposition_tuple" can skip the calculation of the function QR_decomposition()

    :param A: Coefficient matrix
    :type A: np.ndarray
    :param b: Right-hand side vector
    :type b: np.ndarray
    :param QR_decomposition_tuple: Precomputed QR decomposition (A, d)
    :type QR_decomposition_tuple: tuple
    :return: Least squares solution
    :rtype: np.ndarray
    """
    if QR_decomposition_tuple != None:
        A,d = QR_decomposition_tuple 
    else :
        A,d = QR_decomposition(A)
    m, n = A.shape
    for j in range(n):
        if j == m-1 :
            break
        v = np.zeros((m,1))
        v[j,0]=1
        v[j+1:,0]=A[j+1:,j]
        beta=d[j,0]
        b = (np.eye(m)-beta*v@v.T)@b
    return p1_1.solve_LUx_b(np.eye(n),A[:n,:n],b)

def p3_1():
    A, b=p1_1.create_matrix_p1_1()
    L, U=p1_1.gauss_elimation(A)
    Ans1_1=p1_1.solve_LUx_b(L, U, b)
    L, U, P=p1_1.gauss_elimation(A, method='column')
    Ans1_2=p1_1.solve_LUx_b(L, U, P.dot(b))
    Ans1_3=QR_solve_Ax_b(A,b)
    # print(Ans1_3)
    p2_2.show_graph(
        {
            "x":np.array([n for n in range(84)]),
            "Default Gauss":abs(Ans1_1),
            "Column Gauss":abs(Ans1_2),
            "QR Decomposition":abs(Ans1_3),
            "1":np.array([1 for n in range(84)])
        },
        title="Ans for equations in chapter 1 problem 1"
    )
    A, b=p1_2.create_matrix_p1_2_1()
    Ans2_1=p1_2.square_root_method(A,b)
    Ans2_2=p1_2.square_root_method(A,b,True)
    Ans2_3=QR_solve_Ax_b(A,b)
    p2_2.show_graph(
        {
            "x":np.array([n for n in range(len(Ans2_1))]),
            "Square root method":abs(Ans2_1),
            "Advanced square root method":abs(Ans2_2),
            "QR Decomposition":abs(Ans2_3)
        },
        title="Ans for equations in chapter 1 problem 2-1"
    )
    A, b=p1_2.create_matrix_p1_2_2()
    Ans3_1=p1_2.square_root_method(A,b)
    Ans3_2=p1_2.square_root_method(A,b,True)
    Ans3_3=QR_solve_Ax_b(A,b)
    p2_2.show_graph(
        {
            "x":np.array([n for n in range(len(Ans3_1))]),
            "Square root method":abs(Ans3_1),
            "Advanced square root method":abs(Ans3_2),
            "QR Decomposition":abs(Ans3_3),
            "1":np.array([1 for n in range(len(Ans3_1))])
        },
        title="Ans for equations in chapter 1 problem 2-2"
    )


def ez_test():
    A=np.array([[1.,0,0],[1,1,0],[2,2,3]])
    b=np.array([[1.], [2], [10]])
    x=QR_solve_Ax_b(A,b)
    print(x)
    print(A@x-b)
    x=Solution_of_LS_problem(A,b)
    print(x)
    print(A@x-b)

if __name__ == '__main__':
    p3_1()


