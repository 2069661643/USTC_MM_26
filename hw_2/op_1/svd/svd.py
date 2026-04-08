from package import p3_1
import numpy as np

def givens(a,b) -> tuple[float, float]:
    """
    Givens Transformation.

    :param a: input num
    :param b: input num
    :return: (c, s), two number, cos and sin, [[c, s], [-s, c]]
    :rtype: tuple[float, float]
    Matrix transformation: [[c, s], [-s, c]] @ [a, b]^T = [sqrt(a^2+b^2), 0]^T
    """
    r = np.hypot(a, b)
    if r == 0:
        return 1.0, 0.0
    c = a / r
    s = b / r
    return c, s

def two_diagonalization(A:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Two diagonalization algorithm (Alg 7.6.1)

    :param A: input matrix
    :type A: np.ndarray
    :return: (P, B, H), P, H are orthogonal, PAH=B, B is a two diagonal matrix.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    A = np.array(A,dtype=np.float64)
    m, n = A.shape
    P = np.eye(m)
    H = np.eye(n)
    for k in range(n):
        if k < m:
            v, b = p3_1.house(A[k:,k:k+1])
            ut = b * v.T @ A[k:,k:]
            A[k:,k:] = A[k:,k:] - v @ ut
            P[k:,k:] =  (np.eye(m-k) - b * v @ v.T) @ P[k:,k:]
        if k < n-1:
            v, b = p3_1.house(A[k:k+1,k+1:].T)
            u = b * A[k:,k+1:] @ v
            A[k:,k+1:] = A[k:,k+1:] - u @ v.T
            H[k+1:,k+1:] = H[k+1:,k+1:] @ ( np.eye(n-k-1) - b * v @ v.T )
    return P, A, H

def svd_iter_step(B:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One iteration of SVD algorithm with Wilkinson shift (Alg 7.6.2)
    :param B: input matrix
    :type B: np.ndarray
    :return: (P, S, Q), P, Q are orthogonal, PBQ = S, S is a two diagonal matrix.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    B = np.array(B,dtype=np.float64)
    n = B.shape[0]
    if n == 1:
        return np.eye(1),B,np.eye(1)
    if n == 2:
        return two_order_svd(B)
    P = np.eye(n)
    Q = np.eye(n)

    # parameters in alg 7.6.2
    delta = [B[i, i] for i in range(n)]
    gamma = [B[i, i+1] for i in range(n-1)]
    a = delta[n-1] * delta[n-1] + gamma[n-2] * gamma[n-2]
    de = (delta[n-2] * delta[n-2] + gamma[n-3] * gamma[n-3] - a) / 2
    b = delta[n-2] * gamma[n-2]
    nu = a - b * b / (de + np.sign(de) * np.sqrt(de*de + b*b))

    y = delta[0] * delta[0] - nu
    z = delta[0] * gamma[0]

    for k in range(n-1):
        # right givens B = BG
        c, s = givens(y, z)
        G = np.array([[c, -s], [s, c]])
        B[k:k+2,k:k+2] = B[k:k+2,k:k+2] @ G
        Q[:,k:k+2] = Q[:,k:k+2] @ G
        y = B[k,k]
        z = B[k+1,k]

        # left givens B = GB
        c, s = givens(y, z)
        G = np.array([[c, s], [-s, c]])
        B[k:k+2,k:k+2] = G @ B[k:k+2,k:k+2]
        P[k:k+2,:] = G @ P[k:k+2,:]
        if k < n-2:
            y = B[k,k+1]
            z = B[k,k+2]
    return P,B,Q

def two_order_svd(A:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two order svd algorithm.
    :param A: input matrix, shape = (2,2)
    :type A: np.ndarray
    :return: (U, S, V), U, V are orthogonal UAV = S
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    A = np.array(A,dtype=np.float64)
    # Compute elements of A^T A
    E = A[0,0]**2 + A[1,0]**2
    F = A[0,0]*A[0,1] + A[1,0]*A[1,1]
    G = A[0,1]**2 + A[1,1]**2
    
    # Calculate eigenvectors of A^T A (which form V)
    theta = 0.5 * np.arctan2(2*F, E - G)
    c, s = np.cos(theta), np.sin(theta)
    V = np.array([[c, -s], 
                  [s,  c]])
    
    # Singular values squared (directly compute without storing in list)
    s1_sq = E*c**2 + 2*F*c*s + G*s**2
    s2_sq = E*s**2 - 2*F*c*s + G*c**2
    s1 = np.sqrt(max(0.0, s1_sq))
    s2 = np.sqrt(max(0.0, s2_sq))
    
    # Compute U from AV = US
    Av1 = A @ V[:, 0]
    Av2 = A @ V[:, 1]
    
    u1 = Av1 / s1 if s1 > 1e-14 else np.array([1.0, 0.0])
    u2 = Av2 / s2 if s2 > 1e-14 else np.array([-u1[1], u1[0]])
    
    U = np.vstack([u1, u2])
    S = np.diag([s1, s2])
    
    return U, S, V

def clear_one_row(B:np.ndarray, i:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Step 4(i)

    :param B: input matrix
    :type B: np.ndarray
    :param i: the row index
    :type i: int
    :return: (G, S) where GB = S
    """
    B = np.array(B,dtype=np.float64)
    if B[i,i] != 0:
        raise ValueError("B[i,i] is not 0, i = %d\nmat = %s"%(i,B))
    if B[i,i+1] == 0:
        return np.eye(B.shape[0]), B
    n = B.shape[0]
    P = np.eye(n)
    for j in range(i+1, B.shape[0]):
        c,s = givens(B[i,j], B[j,j])
        G = np.array([[-s, c], [c, s]])
        B[[i,j],j:] = G @ B[[i,j],j:]
        P[[i, j], :] = G @ P[[i, j], :]
    return P,B

def svd(A:np.ndarray,
        eps:float = 1e-8
        ) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD algorithm (Alg 7.6.3)

    :param A: input matrix
    :type A: np.ndarray
    :param eps: error constant
    :type eps: float
    :return: (U, S, V), A = U @ S @ V
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    A = np.array(A,dtype=np.float64)

    # Step 1
    m, n = A.shape
    transpose_flag = False
    if m < n:
        A = A.T
        m, n = n, m
        transpose_flag = True

    # Step 2
    P, B, Q = two_diagonalization(A)
    B = B[:n,:n]

    while True:
        # Step 3(i)
        for i in range(n-1):
            if abs(B[i, i+1]) < eps * (abs(B[i, i]) + abs(B[i + 1, i + 1])):
                B[i, i + 1] = 0

        # Step 3(ii)
        b_norm = np.linalg.norm(B,ord=np.inf)
        for i in range(n):
            if B[i, i] < eps * b_norm:
                B[i, i] = 0

        # Step 3(iii)
        q = n
        for i in range(n-2,-1,-1):
            if B[i, i+1] == 0:
                q = i+1
            else:
                break
        p = q
        for i in range(q-2,-1,-1):
            if i == n-1:
                p = n-1
            elif B[i, i+1] != 0:
                p = i
            else:
                break

        # Step 3(iv)
        if q <= 1:
            break

        # Step 4(i)
        B22 = B[p:q,p:q]
        bn = B22.shape[0]
        goto_step3 = False
        for i in range(bn-2,-1,-1):
            if B22[i, i] == 0:
                G, B22 = clear_one_row(B22, i)
                P[p:q,:] = G @ P[p:q,:]
                B[p:q, p:q] = B22

                # Step 2
                P_, B, Q_ = two_diagonalization(B)
                P = P_ @ P
                Q = Q @ Q_
                goto_step3 = True
                break
        if goto_step3:
            continue

        # Step 4(ii)
        U, B22, V = svd_iter_step(B22)
        P[p:q, :] = U @ P[p:q, :]
        B[p:q, p:q] = B22
        Q[:, p:q] = Q[:, p:q] @ V

    # Step 5
    # Sort the singular values in descending order
    diag_B = np.diag(B)
    sort_order = np.argsort(diag_B)[::-1]

    B = B[sort_order, :][:, sort_order]
    P = P[sort_order, :]
    Q = Q[:, sort_order]

    # Step 6
    # Transform the matrix into the form in Docstring
    if transpose_flag:
        U = Q
        V = P
        S = np.zeros((n, m))
        np.fill_diagonal(S, np.diag(B))
    else:
        U = P.T
        V = Q.T
        S = np.zeros((m, n))
        np.fill_diagonal(S, np.diag(B))

    return U, S, V

def ez_test():
    A = np.array([[1, 2, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 9]])
    U, S, V = svd(A)
    print("A=")
    print(A)
    print("U=")
    print(U)
    print("S=")
    print(S)
    print("V=")
    print(V)
    print("U S V=")
    print(U @ S @ V)

if __name__ == "__main__":
    ez_test()