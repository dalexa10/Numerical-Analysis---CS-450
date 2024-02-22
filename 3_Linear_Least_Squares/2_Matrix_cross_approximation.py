import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt

np.random.seed(450) # fix the seed

m=100
n=100

S = np.diag(4**np.linspace(0,n,n))
U,_ = la.qr(np.random.randn(m*n).reshape((m,n)))
V,_ = la.qr(np.random.randn(n*n).reshape((n,n)))
A1 = U@S@V
A1 = n*n*A1 /la.norm(A1)
S = np.diag(np.linspace(0,n,n))
U,_ = la.qr(np.random.randn(m*n).reshape((m,n)))
V,_ = la.qr(np.random.randn(n*n).reshape((n,n)))
A2 = U@S@V
A2 = n*A2 /la.norm(A2)
perm = np.random.choice(n,n,replace=False)
A = np.zeros((m,n)) # input for solution
k = 10
A[:,perm[:k]] = A2[:,:k]
A[:,perm[k:]] = A1[:,k:]

ranks = np.arange(min(n,m)//10, min(n,m)//4, 1) # input for solution

#%% Solution

q, r, p = sla.qr(A, pivoting=True)
q_T, r_T, p_T = sla.qr(A.T, pivoting=True)
U, s, V_T = sla.svd(A)

# Initialization outpus np arrays
error_cross = np.zeros(ranks.shape[0])
error_svd = np.zeros(ranks.shape[0])
error_firstk = np.zeros(ranks.shape[0])
volume_cross = np.zeros(ranks.shape[0])
volume_firstk = np.zeros(ranks.shape[0])

for i, r in enumerate(ranks):
    # Cross approximation pivoting r
    A_r_cross = A[:,p[:r]] @ sla.inv(A[p_T[:r], :][:, p[:r]]) @ A[p_T[:r], :]
    #'Optimal SVD'
    A_r_SVD = U[:, :r] @ np.diag(s[:r]) @ V_T[:r, :]
    # Cross approximation first k
    A_k_cross = A[:,:r] @ sla.inv(A[:r, :][:, :r]) @ A[:r, :]
    # Error cross pivoting
    error_cross[i] = sla.norm(A - A_r_cross)
    error_svd[i] = sla.norm(A - A_r_SVD)
    error_firstk[i] = sla.norm(A - A_k_cross)
    # Volume calculation
    volume_cross[i] = np.abs(sla.det(A[p_T[:r], :][:, p[:r]]))
    volume_firstk[i] = np.abs(sla.det(A[:r, :][:, :r]))

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(ranks, error_cross, label='Cross approximation')
ax[0].plot(ranks, error_svd, label='SVD approximation')
ax[0].plot(ranks, error_firstk, label='First k approximation')
ax[0].set_xlabel('Rank')
ax[0].set_yscale('log')
ax[0].set_ylabel('Error')
ax[0].set_title('Error')
ax[0].legend()

ax[1].plot(ranks, volume_cross, label='Cross approximation')
ax[1].plot(ranks, volume_firstk, label='First k approximation')
ax[1].set_xlabel('Rank')
ax[1].set_yscale('log')
ax[1].set_yscale('log')
ax[1].set_ylabel('Volume')
ax[1].set_title('Volume')
ax[1].legend()

plt.tight_layout()
plt.show()