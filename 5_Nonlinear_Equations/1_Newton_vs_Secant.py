import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

def f(x):
    """ Function to find the root of """
    return x**2 * (x - 10)

def gradf(x):
    """ Gradient of the function """
    #TODO implement automatic differentiation with autograd or jax
    # so far, this is done manually
    return 2*x*(x-10) + x**2

def newton(x0, f, gradf, r, knewton):
    """ Iterative method to find the root of a function using Newton's method"""

    newton_iterates = []
    error_nw_ls = []
    newton_iterates.append(x0)
    error_nw_ls.append(np.abs((x0 - 10)/10))

    for k in range(knewton):
        x = newton_iterates[k]
        x = x - f(x)/gradf(x)
        newton_iterates.append(x)
        error_nw_ls.append(np.abs((x - 10)/10))

    newton_iterates = np.array(newton_iterates)

    return newton_iterates, newton_iterates[-1], error_nw_ls

def secant(x0, x1, f, r, ksecant):
    """ Iterative method to find the root of a function using the secant method """

    secant_iterates = []
    error_sec_ls = []
    secant_iterates.extend([x0, x1])
    error_sec_ls.extend([np.abs((x0 - 10)/10), np.abs((x1 - 10) / 10)])

    for k in range(ksecant-1):
        x0 = secant_iterates[-2]
        x1 = secant_iterates[-1]
        x = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        secant_iterates.append(x)
        error_sec_ls.append(np.abs((x - 10)/10))

    secant_iterates = np.array(secant_iterates)

    return secant_iterates, secant_iterates[-1], error_sec_ls

if __name__ == '__main__':

    # Inputs
    x0 = 25
    x1 = 15
    r = 10 ** (-13.)
    knewton = 8
    ksecant = 10

    # call routines
    newton_iterates, newton_zero, newton_error_ls = newton(x0, f, gradf, r, knewton)
    secant_iterates, secant_zero, secant_error_ls = secant(x0, x1, f, r, ksecant)

    # compute and plot error
    fig, ax = plt.subplots()
    ax.plot(newton_error_ls, label='Newton method')
    ax.plot(secant_error_ls, label='Secant method')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    ax.set_title('Error vs iteration')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

