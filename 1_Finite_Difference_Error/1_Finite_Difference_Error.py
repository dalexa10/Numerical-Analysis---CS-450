import numpy as np
import random
import matplotlib.pyplot as plt

k = random.random() * 4 + 3
n_values = [2**n_exp for n_exp in range(5, 20)]


def f(x):
    return np.sin(k * x)

def set_indexes(n):
    idx = np.arange(n)
    idx_p1 = np.roll(idx, -1)
    idx_m1 = np.roll(idx, 1)
    return idx_p1, idx_m1

abs_err_values = np.zeros(len(n_values), dtype='float64')
truncation_err_values = np.zeros(len(n_values), dtype='float64')
rounding_err_values = np.zeros(len(n_values), dtype='float64')

idx_control = 0
for n in n_values:
    x = np.linspace(0, 1, n)
    eps = np.finfo(np.float64).eps
    idx_p1, idx_m1 = set_indexes(n)
    h = 1 / (n - 1)

    # Compute the absolute error values of interior points
    f_2prime_ap = (f(x[idx_p1]) - 2 * f(x) + f(x[idx_m1])) / h**2
    f_2_exact = - k**2 * np.sin(k * x)
    abs_err = np.amax(np.abs(f_2prime_ap[1: -1] - f_2_exact[1: -1]))
    abs_err_values[idx_control] = abs_err

    # Compute the truncation error upper bound
    truncation_err = np.amax(h**2 * np.abs(k**4) / 12)
    truncation_err_values[idx_control] = truncation_err

    # Compute the rounding error bound
    rounding_err = np.amax(4 * eps * np.abs(k) / h**2)
    rounding_err_values[idx_control] = rounding_err

    idx_control += 1

# ------------ plotting code below, no need to change
h_values = []
for n in n_values:
    h_values.append(1 / (n - 1))
h_values = np.array(h_values)

plt.xlabel(r"$h$")
plt.loglog(h_values, truncation_err_values + rounding_err_values, label=r"Predicted Error Bound")
plt.loglog(h_values, abs_err_values, label="Computed Absolute Error")
plt.legend(loc='lower right')
plt.show()