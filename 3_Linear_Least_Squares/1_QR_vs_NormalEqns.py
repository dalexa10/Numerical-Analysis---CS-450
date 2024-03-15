import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def matrix_generator(m, n, cond):
    np.random.seed(0)
    A = np.random.randn(m, n)
    U, _, Vt = la.svd(A)
    S = np.diag(np.linspace(1, cond, min(m, n)))
    A = U @ S @ Vt
    b = A @ np.random.randn(n)
    x_accurate = la.solve(A, b)
    return A, b, x_accurate
def condition_number(A):
    _, s, _ = la.svd(A)
    return s[0] / s[-1]

x_normal_list = []
x_qr_list = []
r_normal_list = []
r_qr_list = []
cond_A_list = []

cond_ls = [10**(i/2) for i in range(1, 11)]
A_list = [matrix_generator(100, 100, cond_ls[i])[0] for i in range(10)]
b_list = [matrix_generator(100, 100, cond_ls[i])[1] for i in range(10)]
x_accurate_list = [matrix_generator(100, 100, cond_ls[i])[2] for i in range(10)]


for A, b, x_accurate in zip(A_list, b_list, x_accurate_list):

    # Add code here
    x_normal = la.solve(A.T@A, A.T@b)
    Q, R = np.linalg.qr(A)
    x_qr = la.solve(R,Q.T@b)
    r_normal = la.norm(x_normal - x_accurate) / la.norm(x_accurate)
    r_qr = la.norm(x_qr - x_accurate) / la.norm(x_accurate)
    cond_A = condition_number(A)

    x_normal_list.append(x_normal)
    x_qr_list.append(x_qr)
    r_normal_list.append(r_normal)
    r_qr_list.append(r_qr)
    cond_A_list.append(cond_A)

# Add plot code here
fig, ax = plt.subplots()
ax.plot(cond_A_list, r_normal_list, label='Normal Equations')
ax.plot(cond_A_list, r_qr_list, label='QR')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('rel. err.')
ax.set_xlabel(r'$\kappa(A)$')
ax.set_title('Relative error vs Condition number')
ax.legend(loc='best')
plt.show()

print('It can be noticed that as the condition number kappa(A) increases, '
      'the relative error of the solution increase as well. This is because a high '
      'condition number is an indicative that a matrix (and its inverse) are close to '
      'be singular and hence, the solvers incurr in more relative error when attempting to '
      'solve the system. It is also noted that the normal equations present a higher relative '
      'error than the QR method for a same condition number kappa(A). This is because the condition '
      'number of A.T@A is the square of the A matrix itself, which means that the matrix A.T@A is '
      'even closer to be singular than A itself. On the other hand, QR is a factorization that '
      'preserve the Euclidean norm (i.e. does not change the least square solution) and does not amplify the error')