import numpy as np
from . import p2_1, p1_1
import random 
import matplotlib.pyplot as plt

def create_matrix_p2_2(n):
    """
    Return A, x ~ (U[0,1], ..., U[0,1])

    :param n: Matrix dimension
    :type n: int
    :return: (A, x) where A is coefficient matrix and x is random vector with uniform distribution
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    A = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i == j :
                A[i,j]=1
            elif i > j :
                A[i,j]=-1
            elif j == n-1 :
                A[i,j]=1
    x = np.zeros([n,1])
    for i in range(n):
        x[i] = random.random()
    return A, x

def show_graph(ans_dict:dict,log=False,title=None,xlabel='X-axis',ylabel='Y-axis'):
    """
    Show a pic to compare those parameters
    The value of ans_dict should be a list[number]-like object

    :param ans_dict: Dictionary containing data to plot
    :type ans_dict: dict
    :param log: Use logarithmic scale for y-axis
    :type log: bool
    :param title: Plot title
    :type title: str
    :param xlabel: X-axis label
    :type xlabel: str
    :param ylabel: Y-axis label
    :type ylabel: str
    :return: None
    :rtype: None
    """
    if "x" in ans_dict:
        for key in ans_dict:
            if key != "x":
                n = len(ans_dict[key])
                plt.plot(np.array(ans_dict["x"]).flatten(), ans_dict[key], label=key)
    else:    
        for key in ans_dict:
            n = len(ans_dict[key])
            plt.plot([x for x in range(n)], ans_dict[key], label=key)

    if log:
        plt.yscale('log')
    if title == None:
        plt.title('Ans')
    else:
        plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()

def p2_2():
    error_ans_list=[]
    error_expected_list=[]
    n_list=[]
    for n in range(5,31):
        A, x=create_matrix_p2_2(n)
        b = A@x
        L, U, P=p1_1.gauss_elimation(A,'column')
        x_ans=p1_1.solve_LUx_b(L,U,P@b)
        error_ans=np.max(abs(x_ans-x))/np.max(x)
        error_expected=np.max(abs(b-A@x_ans))*np.linalg.norm(A, ord=np.inf)*p2_1.climbing_blind_estimation_inverse(A)/np.max(abs(b))
        error_ans_list.append(error_ans)
        error_expected_list.append(error_expected)
        n_list.append(n)
    show_graph({
        "Actual Error":np.array(error_ans_list),
        "Estimated Error":np.array(error_expected_list),
        "x":np.array(n_list)
                      })
    show_graph({
        "Estimated Error/Actual Error":np.array(error_expected_list)/np.array(error_ans_list),
        "1":np.ones(31-5),
        "x":np.array(n_list)
                      },False)
    print({
        "Actual Error":np.array(error_ans_list),
        "Estimated Error":np.array(error_expected_list)
    })

if __name__ == "__main__":
    p2_2()