import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la


def full_discretization(u, dx, dt):
    """Implementation of full descretization method

    Args:
        u (ndarray): 1D array of floats for initial conditions
        dx (float): delta x
        dt (float): delta t

    Returns:
        u (ndarray): 1D array of floats for the solution
    """
    u = np.copy(u)
    # Your code here
    k = np.arange(0, u.shape[0])  # 0, ..., nx-1
    km1 = np.roll(k, 1)  # nx-1, 0, 1, ..., nx-2
    kp1 = np.roll(k, -1)  # 1, ..., nx

    # Evolve over time
    u_k = u[k]
    u_kp1 = u[kp1]
    u_km1 = u[km1]
    u[:] = u + (dt / dx ** 2) * (u_kp1 - 2 * u_k + u_km1)

    # Impose BC
    u[0] = 0
    u[-1] = 0
    return u


def backward_euler(u, dx, dt):
    """Implementation of backward euler method

    Args:
        u (ndarray): 1D array of floats for initial conditions
        dx (float): delta x
        dt (float): delta t

    Returns:
        u (ndarray): 1D array of floats for the solution
    """
    u = np.copy(u)
    c = dt / dx ** 2
    n = len(u)
    A = - np.diag(np.ones(n - 1), -1) + 2 * np.diag(np.ones(n), 0) - np.diag(np.ones(n - 1), 1)
    A = c * A
    A += np.eye(n)

    u = la.solve(A, u)

    return u


def crank_nicolson(u, dx, dt):
    """Implementation of crank nicolson method

    Args:
        u (ndarray): 1D array of floats for initial conditions
        dx (float): delta x
        dt (float): delta t

    Returns:
        u (ndarray): 1D array of floats for the solution
    """
    # Your code here
    c = dt / (2 * dx ** 2)
    n = len(u)
    A = - np.diag(np.ones(n - 1), -1) + 2 * np.diag(np.ones(n), 0) - np.diag(np.ones(n - 1), 1)
    A = c * A
    A += np.eye(n)

    B = np.diag(np.ones(n - 1), -1) - 2 * np.diag(np.ones(n), 0) + np.diag(np.ones(n - 1), 1)
    B = c * B
    B += np.eye(n)

    u = la.solve(A, B @ u)
    return u


def plot_solution(X, T, U, dt, dx, integrator, name, plt_steps, use_wireframe):
    """Plot solution to the heat equation

    Args:
        X (ndarray): linspace of x (implemented)
        T (ndarray): 1D array of time step reported every plt_steps steps
        U (ndarray): 1D array of solution u reported every plt_steps steps
        dt (float): delta t
        dx (float): delta x
        integrator (callable): the function for computing the solution u
        name (str): name of the method
        plt_steps (int): plot solution for every plt_steps time steps
        use_wireframe (bool): plot with wireframe or not
    """
    # Your code here
    fig, ax = plt.subplots()

    if use_wireframe:
        ax = fig.add_subplot(111, projection='3d')
        X, T = np.meshgrid(X, T)
        ax.plot_surface(X, T, U, cmap='viridis', edgecolor='none')

        ax.set_title(f'{name} method')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        plt.show()

    else:
        for i in range(0, U.shape[0], plt_steps):
            ax.plot(X, U[i], label=f't={T[i]:.2f}')
        ax.legend()

        ax.set_title(f'{name} method')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        plt.show()




def integrate_pde(dt, dx, integrator, name, plt_steps, use_wireframe):
    """Find and plot solution to the heat equation using a given integrator

    * Remember to implement your ploting function!

    Args:
        dt (float): delta t
        dx (float): delta x
        integrator (callable): the function for computing the solution u
        name (str): name of the method
        plt_steps (int): plot solution for every plt_steps time steps
        use_wireframe (bool): plot with wireframe or not

    Returns:
        X (ndarray): linspace of x (implemented)
        T (ndarray): 1D array of time step reported every plt_steps steps
        U (ndarray): 1D array of solution u reported every plt_steps steps
    """
    t = 0.
    endt = 0.06
    x = 0.
    endx = 1.

    nx = int(1.0 / dx + 1)
    X = np.linspace(x, endx, nx)

    u = np.zeros(nx, dtype=float)
    u[:int(nx / 2)] = 2.0 * X[:int(nx / 2)]
    u[int(nx / 2):] = 2 - 2.0 * X[int(nx / 2):]
    U = [u]
    T = [t]

    for i in range(int((endt - t) / dt)):
        u = integrator(u, dx, dt)
        if i % plt_steps == 0 or i == int(endt / dt) - 1:
            U.append(u)
            T.append(i * dt + t)

    T = np.array(T)
    U = np.array(U)

    # plot the solution using student implementation
    plot_solution(X, T, U, dt, dx, integrator, name, plt_steps, use_wireframe)

    return X, T, U


x_full_exp, t_full_exp, u_full_exp = integrate_pde(0.0012, 0.05, full_discretization, 'Explicit', 10, True)
x_full_full, t_full_full, u_full_full = integrate_pde(0.0013, 0.05, full_discretization, 'Full', 10, True)
x_be_05, t_be_05, u_be_05 = integrate_pde(0.005, 0.05, backward_euler, 'Backward Euler dx=0.05', 1, True)
x_cn_05, t_cn_05, u_cn_05 = integrate_pde(0.005, 0.05, crank_nicolson, 'Crank-Nicolson dx=0.05', 1, True)
x_be_005, t_be_005, u_be_005 = integrate_pde(0.005, 0.005, backward_euler, 'Backward Euler dx=0.005', 1, False)
x_cn_005, t_cn_005, u_cn_005 = integrate_pde(0.005, 0.005, crank_nicolson, 'Crank-Nicolson dx=0.005', 1, False)
plt.show()