import numpy as np
import matplotlib.pyplot as plt

def gauss_elimation(A, method='default'):
    """
    Calculate A=LU

    :param A: Input matrix
    :type A: np.array
    :param method: 'default' or 'column'
    :type method: str
    :return: (L, U) or (L, U, P) where L is lower triangular matrix, U is upper triangular matrix, P is permutation matrix (only when method='column')
    :rtype: tuple[np.ndarray, np.ndarray] or tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    if A.shape[0] != A.shape[1] or A.ndim != 2:
        raise ValueError("Invalid A")
    n = A.shape[0]
    A = A.copy()
    if method == 'default':
        L_list = []
        L1_list = []
        for k in range(n-1):
            a = A[k,k]
            if a == 0:
                raise ValueError("The gauss process counter a 0 when k = %d"%(k))
            lk=np.zeros([n,1])
            for j in range(k+1, n):
                lk[j]=-A[j,k]/a
            ek=np.zeros([n,1])
            ek[k,0]=1
            Lk=np.eye(n)+np.dot(lk,ek.T)
            Lk1=np.eye(n)-np.dot(lk,ek.T)
            A=np.dot(Lk,A)
            L_list.append(Lk)
            L1_list.append(Lk1)
        U = A
        L = np.eye(n)
        for k in range(n-1):
            L=np.dot(L1_list[n-2-k], L)
        return L, U
    elif method == 'column':
        L_list = []
        L1_list = []
        S_list = []
        S1_list = []
        for k in range(n-1):
            i_m = np.argmax(abs(A[k:n,k]), axis=0) + k
            a = A[i_m, k]
            if a == 0:
                raise ValueError("The gauss process counter a 0 when k = %d"%(k))
            # S_pq
            Sk = np.eye(n)
            Sk[k,k]=0
            Sk[i_m,k]=1
            Sk[i_m,i_m]=0
            Sk[k,i_m]=1
            Sk1 = Sk.copy()
            S_list.append(Sk)
            S1_list.append(Sk1)
            # L_k
            A = Sk.dot(A)
            lk=np.zeros([n,1])
            for j in range(k+1, n):
                lk[j]=-A[j,k]/a
            ek=np.zeros([n,1])
            ek[k,0]=1
            Lk=np.eye(n)+np.dot(lk,ek.T)
            Lk1=np.eye(n)-np.dot(lk,ek.T)
            A=np.dot(Lk,A)
            L_list.append(Lk)
            L1_list.append(Lk1)
        U = A
        L = np.eye(n)
        P = np.eye(n)
        for k in range(n-1):
            L=np.dot(L1_list[n-2-k], L)
            L=np.dot(S1_list[n-2-k], L)
            P=np.dot(S_list[k], P)
        return P.dot(L), U, P
    else:
        raise ValueError("Method invalid")

    # print(L.dot(U))
    return None

def create_matrix_p1_1():
    """
    Return A, b for p1

    :return: (A, b) where A is coefficient matrix and b is right-hand side vector
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    A=6*np.eye(84)
    for i in range(83):
        A[i,i+1]=1.
        A[i+1,i]=8.
    b=np.zeros([84, 1])
    b[0, 0]=7.
    for i in range(1, 83):
        b[i, 0]=15.
    b[83, 0]=14.
    return A, b

def solve_LUx_b(L, U, b):
    """
    Return x from LUx=b

    :param L: Lower triangular matrix
    :type L: np.ndarray
    :param U: Upper triangular matrix
    :type U: np.ndarray
    :param b: Right-hand side vector
    :type b: np.ndarray
    :return: Solution vector
    :rtype: np.ndarray
    """
    n=L.shape[0]
    y=np.zeros([n, 1])
    # Ly=b 
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i,j]*y[j,0]
        y[i,0]=(b[i,0]-sum)/L[i,i]
    x=np.zeros([n, 1])
    # Ux=y
    for i_ in range(n):
        i = n - 1 - i_
        sum = 0
        for j_ in range(i_):
            j = n - 1 - j_
            sum += U[i,j]*x[j,0]
        x[i,0]=(y[i,0]-sum)/U[i,i]
    return x

def ez_test():
    A=np.array([[1,0,0],[1,1,0],[2,2,3]])
    b=np.array([[1], [2], [10]])
    L, U, P=gauss_elimation(A, method='column')
    print(L)
    print(U)
    print(P)

def p1_1():
    A, b=create_matrix_p1_1()
    L, U=gauss_elimation(A)
    Ans1=solve_LUx_b(L, U, b)
    L, U, P=gauss_elimation(A, method='column')
    Ans2=solve_LUx_b(L, U, P.dot(b))
    Ans=np.ones([84,1])
    print(Ans1.T, Ans2.T)
    compare_ans({"Expected":abs(Ans), "Default Gauss":abs(Ans1), "Column Gauss":abs(Ans2)})


def compare_ans(ans_dict:dict,log=True):
    """
    Show a pic to compare those parameters

    :param ans_dict: Dictionary containing data to plot
    :type ans_dict: dict
    :param log: Use logarithmic scale for y-axis
    :type log: bool
    :return: None
    :rtype: None
    """
    if "x" in ans_dict:
        for key in ans_dict:
            if key != "x":
                n = ans_dict[key].size
                plt.plot(ans_dict["x"].flatten(), ans_dict[key].flatten(), label=key)
    else:    
        for key in ans_dict:
            n = ans_dict[key].size
            plt.plot([x for x in range(n)], ans_dict[key].flatten(), label=key)

    if log:
        plt.yscale('log')
    plt.title('Ans')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()
    

if __name__ == "__main__":
    p1_1()