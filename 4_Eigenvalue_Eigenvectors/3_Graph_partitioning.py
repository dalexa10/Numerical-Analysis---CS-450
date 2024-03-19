__author__ = 'Dario Rodriguez'

import numpy as np
import scipy.sparse as ss
import numpy.linalg as la
import matplotlib.pyplot as plt

"""
we consider the Fiedler algorithm, which partitions graphs by finding eigenvectors of a matrix 
associated with the graph. The Lanczos algorithm was implemented to find a particular eigenvector.
"""


# def open(filename, mode="r"):
#     """
#     Only used in the autograder. No need to use this locally.
#     """
#     try:
#         data = data_files["data/"+filename]
#     except KeyError:
#         raise IOError("file not found")
#
#     # 'data' is a 'bytes' object at this point.
#
#     from io import StringIO
#     return StringIO(data.decode("utf-8"))

def readmesh(fname):
    """
    Read a mesh file and return vertics as a (npts, 2)
    numpy array and triangles as (ntriangles, 3) numpy
    array. `npts` is the number of vertices of the mesh
    and `ntriangles` is the number of triangles of the
    mesh.
    """
    fname = 'data/' + fname
    with open(fname, "r") as f:
        npoints = int(next(f))
        points = np.zeros((npoints, 2))

        for i in range(npoints):
            points[i, :] = [float(x) for x in next(f).split()]

        ntriangles = int(next(f))
        triangles = np.zeros((ntriangles, 3), dtype=int)

        for i in range(ntriangles):
            triangles[i, :] = [int(x)-1 for x in next(f).split()]

    return points, triangles

def plotmesh(points, triangles, tricolors = None):
    """
    Given a list of points (shape: (npts, 2)) and triangles
    (shape: (ntriangles, 3)), plot the mesh.
    """
    plt.figure()
    plt.gca().set_aspect('equal')

    if (tricolors is None):
        plt.triplot(points[:, 0], points[:, 1], triangles, "bo-", lw = 1.0)
    else:
        plt.tripcolor(points[:, 0], points[:, 1], triangles, facecolors = tricolors, edgecolors = "k")

    return

def mesh2dualgraph(triangles):
    """
    Calculate the graph laplacian of the dual graph associated
    with the mesh given by numpy array traingles.
    """
    n, m = triangles.shape

    assert(m == 3), "Triangle should have exactly three points !!"

    G = np.zeros((n, n))

    for i, ti in enumerate(triangles):
        for j, tj in enumerate(triangles):
            ## If there is a common edge
            if (len( set(ti) - set(tj) ) == 1):
                G[i, j] = G[j, i] = -1

    for i in range(n):
        G[i, i] = -np.sum(G[i, :])

    return ss.csr_matrix(G)

def lanczos(L, x0, niter):
    """
    Compute the Lanczos algorithm for a given matrix L and initial vector x0.
    Note: L is sparse matrix but symmetric.
    """
    n, m = L.shape
    Q = np.zeros((n, niter + 1))
    T = np.zeros((niter, niter))

    Q[:, 0] = np.zeros(n)
    Q[:, 1] = x0 / np.linalg.norm(x0)
    beta_km1 = 0

    for k in range(1, niter):
        u_k = L @ Q[:, k]
        alpha_k = np.dot(Q[:, k], u_k)
        u_k = u_k - alpha_k * Q[:, k] - beta_km1 * Q[:, k - 1]
        beta_k = np.linalg.norm(u_k)

        if (beta_k == 0):
            print("Lanczos algorithm terminated at iteration ", k, " because beta_k = 0")
            break

        Q[:, k + 1] = u_k / beta_k

        T[k - 1, k - 1] = alpha_k
        T[k, k - 1] = beta_k
        T[k - 1, k] = beta_k

        beta_km1 = beta_k
    T[-1, -1] = Q[:, -1].T @ L @ Q[:, -1]
    Q = Q[:, 1:]

    # ---------------------------------------------
    # # Check orthogonality correct
    # --------------------------------------------
    # L_matrix = L.toarray()
    # print(la.norm(Q.T@L_matrix@Q - T) / la.norm(L_matrix))
    # print(la.norm(Q.T@Q - np.eye(niter)))

    return Q, T

def fiedler_ritz(Q, T):
    """ Compute the Fiedler vector using the Ritz vector from returned tri-diagonal matrix T. and the Arnoldi basis Q."""
    w, v = np.linalg.eig(T)
    sorted_idx = np.argsort(w)
    w_min = w[sorted_idx[1]]  # Select second smallest eigenvalue
    v_min = v[:, sorted_idx[1]]  # Select corresponding eigenvector
    fiedlerVec = Q @ v_min  # Compute the Ritz vector (Fiedler vector)

    return fiedlerVec

def fiedler(G, k):
    """
    Calculate the fiedler vector of the graph Laplacian matrix
    'G' using 'k' niter of Lanczos algorithm.
    """
    n, m = G.shape

    assert (n == m), "Matrix should be square !!"

    x0 = np.linspace(1, n, num = n)

    Q, T = lanczos(G, x0, k)
    fiedlerVec = fiedler_ritz(Q, T)

    partitionVec = np.zeros_like(fiedlerVec)
    mfiedler = np.ma.median(fiedlerVec)

    for i in range(n):
        if (fiedlerVec[i] >= mfiedler):
            partitionVec[i] = 1
        else:
            partitionVec[i] = -1

    return partitionVec


if __name__ == "__main__":

    # Change to mesh.2 to see the partitioning of the graph
    points, triangles = readmesh("mesh.1")

    plotmesh(points, triangles)
    plt.show()

    # Generate the Laplacian of the dual graph of G
    G = mesh2dualgraph(triangles)

    partitionVec = fiedler(G, 150)

    plotmesh(points, triangles, partitionVec)