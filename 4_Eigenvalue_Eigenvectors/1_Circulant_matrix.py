import numpy as np
import numpy.linalg as la

S = np.array([[0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
              [1, 0, 0, 0]])


print('Verify that S.T @ S is the identity matrix:')
print(S.T @ S)

Q, R = la.qr(S)
U, sigma, V = la.svd(S, full_matrices=True)
result = U @ np.diag(sigma) @ V

print('QR decomposition:')
print(Q, R)