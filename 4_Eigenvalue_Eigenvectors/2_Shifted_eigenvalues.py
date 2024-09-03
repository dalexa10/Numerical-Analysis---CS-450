import numpy as np
import matplotlib.pyplot as plt

def get_nth_roots_of_unity(n):
    """
    Compute the nth roots of unity
    :param n: (int) number of roots of unity to compute
    :return:
        roots (np.array): array of nth roots of unity
    """
    roots = np.cos(2 * np.pi * np.arange(n) / n) + 1j * np.sin(2 * np.pi * np.arange(n) / n)

    return roots

def shift_roots(roots, shift):
    """
    Shift the roots of unity by a given shift
    :param roots: (np.array) array of roots of unity
    :param shift: (int) shift value
    :return:
        shifted_roots (np.array): array of shifted roots of unity
    """
    shifted_roots = roots + shift
    return shifted_roots

def compute_inverse_roots(roots):
    """
    Compute the inverse of the roots of unity
    :param roots: (np.array) array of roots of unity
    :return:
        inverse_roots (np.array): array of inverse roots of unity
    """
    # Discard if the root is zero
    roots = roots[np.abs(roots) > 1e-10]

    inverse_roots = 1 / roots

    return inverse_roots


def plot_values_complex_plane(values, values_shifted, values_inv):
    """
    Plot the values in the complex plane
    :param values: (np.array) array of complex values
    :param title: (str) title of the plot
    :return:
        None
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].plot(values.real, values.imag, 'bo')
    ax[0].plot([0], [0], 'ro')  # Plot the origin
    ax[0].set_xlabel('Real part')
    ax[0].set_ylabel('Imaginary part')
    ax[0].set_title('Original')

    ax[1].plot(values_shifted.real, values_shifted.imag, 'bo')
    ax[1].plot([0], [0], 'ro')  # Plot the origin
    ax[1].set_xlabel('Real part')
    ax[1].set_ylabel('Imaginary part')
    ax[1].set_title('Shifted')

    ax[2].plot(values_inv.real, values_inv.imag, 'bo')
    ax[2].plot([0], [0], 'ro')  # Plot the origin
    ax[2].set_xlabel('Real part')
    ax[2].set_ylabel('Imaginary part')
    ax[2].set_title('Inverse')

    return fig, ax


if __name__ == '__main__':
    n = 3
    roots = get_nth_roots_of_unity(n)
    shifts = [1, 1j, -1, -1j]

    for shift in shifts:
        shifted_roots = shift_roots(roots, shift)
        inverse_shifted_roots = compute_inverse_roots(shifted_roots)
        fig, ax = plot_values_complex_plane(roots, shifted_roots, inverse_shifted_roots)
        fig.suptitle(f'Shift: {shift}')
        plt.tight_layout()
        plt.show()


