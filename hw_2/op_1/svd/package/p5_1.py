import numpy as np
from . import p4
import matplotlib.pyplot as plt

def conjugate_gradient_method_solve_Ax_b(A,b,x_0=None,iter_end=1e-7,k_max=None):
    """
    Return x from the given Ax=b
    If `x_0` = None, x_0 will be np.zeros((A.shape[0],1))
    If `k_max` = None, k_max will be A.shape[0]

    :param A: Coefficient matrix
    :type A: np.ndarray
    :param b: Right-hand side vector
    :type b: np.ndarray
    :param x_0: Initial guess
    :type x_0: np.ndarray
    :param iter_end: Convergence threshold
    :type iter_end: float
    :param k_max: Maximum iterations
    :type k_max: int
    :return: Solution vector
    :rtype: np.ndarray
    """
    n = A.shape[0]
    if x_0 == None:
        x_0 = np.zeros((n,1))
    if k_max == None:
        k_max = n
    k = 0
    x = x_0
    r = b - A@x
    rho = r.T@r
    while rho > (iter_end*np.linalg.norm(b,2)) and k < k_max:
        k += 1
        if k == 1:
            p = r
        else:
            beta = rho/rho_p
            p = r + beta*p
        w = A@p
        a = rho/(p.T@w)
        x = x + a*p
        r = r - a*w
        rho_p = rho
        rho = r.T@r
    return x

def p5_1():
    n=20
    h=1/n
    S = p4.tridiagonal_matrix(n-1,-0.25,1+h*h/4,-0.25)
    B = -0.25*np.eye(n-1)
    A = np.zeros(((n-1)*(n-1),(n-1)*(n-1)))
    for i in range(n-1):
        if i == 0:
            A[i*(n-1):(i+1)*(n-1),i*(n-1):(i+1)*(n-1)]=S
        else:
            A[i*(n-1):(i+1)*(n-1),i*(n-1):(i+1)*(n-1)]=S
            A[(i-1)*(n-1):(i)*(n-1),i*(n-1):(i+1)*(n-1)]=B
            A[i*(n-1):(i+1)*(n-1),(i-1)*(n-1):(i)*(n-1)]=B
    b = np.zeros(((n-1)*(n-1),1))
    for i in range(n-1):
        for j in range(n-1):
            row = i+j*(n-1)
            b[row,0]=h*h*np.sin((i+1)*(j+1)*h*h)
            if i == 0:
                b[row,0]+=(((j+1)*h)**2)/4
            if i == n-2:
                b[row,0]+=(1+((j+1)*h)**2)/4
            if j == 0:
                b[row,0]+=(((i+1)*h)**2)/4
            if j == n-2:
                b[row,0]+=(1+((i+1)*h)**2)/4

    x = conjugate_gradient_method_solve_Ax_b(A,b)  
    U = np.round(x.reshape((n - 1, n - 1)),4)

    x = np.linspace(h, 1, n - 1)
    y = np.linspace(h, 1, n - 1)
    X, Y = np.meshgrid(x, y)
    Z = U  

    # 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Surface Plot of $u_{i,j}$')
    print(U)
    plt.show()

if __name__ == "__main__":
    p5_1()