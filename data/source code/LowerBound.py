import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)
N = 200
D = 10
X = np.random.uniform(-5, 5, size=(N, D))

w = np.random.randn(D, 1)
f_true = lambda X: np.sin(X.dot(w))     # shape (N,1)

# Noisy observations
noise_std = 0.1
y = f_true(X) + noise_std * np.random.randn(N, 1)

# Radial Basis Function Kernel
# Parameters are lengthscale and variance
# Larger the length scale, smoother the function
# Larger variance means larger vertical scale of the output 
def rbf_kernel(X1, X2, lengthscale, variance):
    X1s = X1 / lengthscale # (N1, D)
    X2s = X2 / lengthscale # (N2, D)
    sq_dists = (
        np.sum(X1s**2, axis=1)[:,None] #(N1,1)
      + np.sum(X2s**2, axis=1)[None,:] #(1,N2) # with broadcasting it makes (N1,N2) 
      - 2*X1s.dot(X2s.T) # (N1,N2) 
    )
    return variance * np.exp(-0.5 * sq_dists)

# Compute the variational bound (eq. 9 in paper)
# sigma2 is the noise added to the final output y
# We DO NOT optimize pseudo-inputs X_m.
def variational_bound(params, X, y, m_idx):
    lengthscale, variance, sigma2 = params
    n = X.shape[0]
    # Inducing inputs
    if m_idx: 
        Xm = X[m_idx]
        #Add the Identity matrix * 1e-6 to prevent the Kmm to be singular matrix
        Kmm = rbf_kernel(Xm, Xm, lengthscale, variance) + 1e-6 * np.eye(len(Xm))
        Kmn = rbf_kernel(Xm, X, lengthscale, variance)
        Z = np.linalg.solve(Kmm, Kmn) #Kmm^(-1)*Kmn
        Qnn = Kmn.T.dot(Z)
        # Trace term: Tr(Knn) - Tr(Qnn)
        trace_Knn = variance * n
        trace_Q = np.trace(Qnn)
        trace_term = trace_Knn - trace_Q
        # Build covariance C = Qnn + sigma2 I for likelihood
        C = Qnn + sigma2 * np.eye(n)
    else: #No inducing points 
        trace_term = variance * n
        C = sigma2 * np.eye(n)
    # Log marginal likelihood part
    L = np.linalg.cholesky(C + 1e-6 * np.eye(n))
    #Since in the log likelihood of Gaussian, we have y'C^(-1)y term.
    #LL^t * alpha = y, alpha = C^(-1)*y
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y[:, 0]))
    #log det(C) = log (det(L)^2) 
    log_det = 2 * np.sum(np.log(np.diag(L)))
    log_lik = -0.5 * (n * np.log(2 * np.pi) + log_det + y[:, 0].dot(alpha))
    # Variational lower bound
    F = log_lik - 0.5 / sigma2 * trace_term
    return F


'''
We now neeed to select training input which maximizes the lower bound and update parameters
'''
