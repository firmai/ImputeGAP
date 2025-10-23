#!/usr/bin/env python3
"""
Simplified MISSNET Imputation Script

This script provides a standalone implementation of the MISSNET algorithm for time series imputation.
MISSNET: Mining of Switching Sparse Networks for Missing Value Imputation in Multivariate Time Series

Based on the paper:
Kohei Obata, Koki Kawabata, Yasuko Matsubara, and Yasushi Sakurai. 2024. 
Mining of Switching Sparse Networks for Missing Value Imputation in Multivariate Time Series. 
In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24).
"""

import warnings
import time
import numpy as np
import pandas as pd
import pickle
import shutil
import os
from typing import Optional, Tuple


def interpolate_matrix(X, how='linear'):
    """Interpolate missing values in a matrix."""
    initial_X = pd.DataFrame(X).interpolate(method=how)
    initial_X = initial_X.ffill()
    initial_X = initial_X.bfill()
    return np.array(initial_X)


def make_dir(input_dir, delete=False):
    """Create directory, optionally deleting existing one."""
    if os.path.isdir(input_dir):
        if delete:
            shutil.rmtree(input_dir)
            os.makedirs(input_dir)
    else:
        os.makedirs(f"{input_dir}")


def empirical_covariance(X, assume_centered=False):
    """Compute empirical covariance matrix."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.ndim == 2:
        if assume_centered:
            covariance = np.dot(X.T, X) / X.shape[0]
        else:
            covariance = np.cov(X.T, bias=1)

    if X.ndim == 3:
        covariance = 0
        for i in range(X.shape[1]):
            X_temp = X[:,i,:]
            if assume_centered:
                covariance += np.dot(X_temp.T, X_temp) / X_temp.shape[0]
            else:
                covariance += np.cov(X_temp.T, bias=1)
        covariance /= X.shape[1]

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance


def prox_laplacian(a, lamda):
    """Prox for l_2 square norm, Laplacian regularisation."""
    return a / (1 + 2. * lamda)


def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x."""
    x = np.ravel(x, order='K')
    if np.issubdtype(x.dtype, np.integer):
        warnings.warn('Array type is integer, np.dot may overflow. '
                      'Data should be float type to avoid this issue',
                      UserWarning)
    return np.dot(x, x)


def init_precision(emp_cov, mode='empirical'):
    """Initialize precision matrix."""
    if isinstance(mode, np.ndarray):
        return mode.copy()

    if mode == 'empirical':
        n_times, _, n_features = emp_cov.shape
        covariance_ = emp_cov.copy()
        covariance_ *= 0.95
        K = np.empty_like(emp_cov)
        for i, (c, e) in enumerate(zip(covariance_, emp_cov)):
            c.flat[::n_features + 1] = e.flat[::n_features + 1]
            K[i] = np.linalg.pinv(c, hermitian=True)
    elif mode == 'zeros':
        K = np.zeros_like(emp_cov)

    return K


def fast_logdet(A):
    """Compute log(det(A)) for A symmetric."""
    sign, ld = np.linalg.slogdet(A)
    if not sign > 0:
        return -np.inf
    return ld


def logl(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)


def loss(S, K, n_samples=None):
    """Loss function for time-varying graphical lasso."""
    if n_samples is None:
        n_samples = np.ones(S.shape[0])
    return sum(
        -ni * logl(emp_cov, precision)
        for emp_cov, precision, ni in zip(S, K, n_samples))


def l1_norm(precision):
    """L1 norm."""
    return np.abs(precision).sum()


def l1_od_norm(precision):
    """L1 norm off-diagonal."""
    return l1_norm(precision) - np.abs(np.diag(precision)).sum()


def objective(n_samples, S, K, Z_0, Z_1, Z_2, alpha, beta, psi):
    """Objective function for time-varying graphical lasso."""
    obj = loss(S, K, n_samples=n_samples)

    if isinstance(alpha, np.ndarray):
        obj += sum(l1_od_norm(a * z) for a, z in zip(alpha, Z_0))
    else:
        obj += alpha * sum(map(l1_od_norm, Z_0))

    if isinstance(beta, np.ndarray):
        obj += sum(b[0][0] * m for b, m in zip(beta, map(psi, Z_2 - Z_1)))
    else:
        obj += beta * sum(map(psi, Z_2 - Z_1))

    return obj


def prox_logdet(a, lamda):
    """Time-varying latent variable graphical lasso prox."""
    es, Q = np.linalg.eigh(a)
    xi = (-es + np.sqrt(np.square(es) + 4. / lamda)) * lamda / 2.
    return np.linalg.multi_dot((Q, np.diag(xi), Q.T))


def soft_thresholding(a, lamda):
    """Soft-thresholding."""
    return np.sign(a) * np.maximum(np.abs(a) - lamda, 0)


def update_rho(rho, rnorm, snorm, iteration=None, mu=10, tau_inc=2, tau_dec=2):
    """Update rho parameter for ADMM."""
    if rnorm > mu * snorm:
        return tau_inc * rho
    elif snorm > mu * rnorm:
        return rho / tau_dec
    return rho


def time_graphical_lasso(
        emp_cov, alpha=0.01, rho=1, beta=1, max_iter=100, n_samples=None,
        verbose=False, psi='laplacian', tol=1e-4, rtol=1e-4,
        return_history=False, return_n_iter=True, mode='admm',
        compute_objective=True, stop_at=None, stop_when=1e-4,
        update_rho_options=None, init='empirical', init_inv_cov=None):

    psi = squared_norm
    prox_psi = prox_laplacian
    psi_node_penalty = False

    Z_0 = init_precision(emp_cov, mode=init)
    if isinstance(init_inv_cov, np.ndarray):
        Z_0[0,:,:] = init_inv_cov

    Z_1 = Z_0.copy()[:-1]
    Z_2 = Z_0.copy()[1:]

    U_0 = np.zeros_like(Z_0)
    U_1 = np.zeros_like(Z_1)
    U_2 = np.zeros_like(Z_2)

    Z_0_old = np.zeros_like(Z_0)
    Z_1_old = np.zeros_like(Z_1)
    Z_2_old = np.zeros_like(Z_2)

    # divisor for consensus variables, accounting for two less matrices
    divisor = np.full(emp_cov.shape[0], 3, dtype=float)
    divisor[0] -= 1
    divisor[-1] -= 1

    if n_samples is None:
        n_samples = np.ones(emp_cov.shape[0])

    checks = []
    for iteration_ in range(max_iter):
        # update K
        A = Z_0 - U_0
        A[:-1] += Z_1 - U_1
        A[1:] += Z_2 - U_2
        A /= divisor[:, None, None]
        A += A.transpose(0, 2, 1)
        A /= 2.

        A *= -rho * divisor[:, None, None] / n_samples[:, None, None]
        A += emp_cov

        K = np.array(
            [
                prox_logdet(a, lamda=ni / (rho * div))
                for a, div, ni in zip(A, divisor, n_samples)
            ])

        if isinstance(init_inv_cov, np.ndarray):
            K[0,:,:] = init_inv_cov

        # update Z_0
        A = K + U_0
        A += A.transpose(0, 2, 1)
        A /= 2.
        Z_0 = soft_thresholding(A, lamda=alpha / rho)

        # other Zs
        A_1 = K[:-1] + U_1
        A_2 = K[1:] + U_2
        if not psi_node_penalty:
            prox_e = prox_psi(A_2 - A_1, lamda=2. * beta / rho)
            Z_1 = .5 * (A_1 + A_2 - prox_e)
            Z_2 = .5 * (A_1 + A_2 + prox_e)
        else:
            Z_1, Z_2 = prox_psi(
                np.concatenate((A_1, A_2), axis=1), lamda=.5 * beta / rho,
                rho=rho, tol=tol, rtol=rtol, max_iter=max_iter)

        if isinstance(init_inv_cov, np.ndarray):
            Z_0[0,:,:] = init_inv_cov
            Z_1[0,:,:] = init_inv_cov

        # update residuals
        U_0 += K - Z_0
        U_1 += K[:-1] - Z_1
        U_2 += K[1:] - Z_2

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
            squared_norm(K - Z_0) + squared_norm(K[:-1] - Z_1) +
            squared_norm(K[1:] - Z_2))

        snorm = rho * np.sqrt(
            squared_norm(Z_0 - Z_0_old) + squared_norm(Z_1 - Z_1_old) +
            squared_norm(Z_2 - Z_2_old))

        obj = objective(
            n_samples, emp_cov, Z_0, K, Z_1, Z_2, alpha, beta, psi) \
            if compute_objective else np.nan

        Z_0_old = Z_0.copy()
        Z_1_old = Z_1.copy()
        Z_2_old = Z_2.copy()

        if verbose:
            print(
                "obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                "eps_pri: %.4f, eps_dual: %.4f" % (obj, rnorm, snorm,
                np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * max(
                    np.sqrt(
                        squared_norm(Z_0) + squared_norm(Z_1) + squared_norm(Z_2)),
                    np.sqrt(
                        squared_norm(K) + squared_norm(K[:-1]) +
                        squared_norm(K[1:]))),
                np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * rho *
                np.sqrt(squared_norm(U_0) + squared_norm(U_1) + squared_norm(U_2))))

        checks.append(obj)
        if stop_at is not None:
            if abs(check.obj - stop_at) / abs(stop_at) < stop_when:
                break

        # convergence check
        e_pri = np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * max(
            np.sqrt(squared_norm(Z_0) + squared_norm(Z_1) + squared_norm(Z_2)),
            np.sqrt(squared_norm(K) + squared_norm(K[:-1]) + squared_norm(K[1:])))
        e_dual = np.sqrt(K.size + 2 * Z_1.size) * tol + rtol * rho * \
            np.sqrt(squared_norm(U_0) + squared_norm(U_1) + squared_norm(U_2))

        if rnorm <= e_pri and snorm <= e_dual:
            break

        rho_new = update_rho(
            rho, rnorm, snorm, iteration=iteration_,
            **(update_rho_options or {}))
        # scaled dual variables should be also rescaled
        U_0 *= rho / rho_new
        U_1 *= rho / rho_new
        U_2 *= rho / rho_new
        rho = rho_new

    else:
        warnings.warn("Objective did not converge.")

    covariance_ = np.array([np.linalg.pinv(x, hermitian=True) for x in Z_0])
    return_list = [Z_0, covariance_]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_ + 1)
    return return_list


class TVGL:
    """Time-Varying Graphical Lasso implementation."""
    
    def __init__(self, alpha=0.01, beta=1., mode='admm', rho=1., tol=1e-4, rtol=1e-4, 
                 psi='laplacian', max_iter=100, verbose=False, assume_centered=False, 
                 return_history=False, update_rho_options=None, compute_objective=True, 
                 stop_at=None, stop_when=1e-4, suppress_warn_list=False, init='empirical'):
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.rho = rho
        self.tol = tol
        self.rtol = rtol
        self.psi = psi
        self.max_iter = max_iter
        self.verbose = verbose
        self.assume_centered = assume_centered
        self.return_history = return_history
        self.update_rho_options = update_rho_options
        self.compute_objective = compute_objective
        self.stop_at = stop_at
        self.stop_when = stop_when
        self.suppress_warn_list = suppress_warn_list
        self.init = init

    def _fit(self, emp_cov, n_samples, init_inv_cov=None):
        """Fit the TimeGraphicalLasso model to X."""
        out = time_graphical_lasso(
            emp_cov, alpha=self.alpha, rho=self.rho, beta=self.beta, mode=self.mode, 
            n_samples=n_samples, tol=self.tol, rtol=self.rtol, psi=self.psi, 
            max_iter=self.max_iter, verbose=self.verbose, return_n_iter=True, 
            return_history=self.return_history, update_rho_options=self.update_rho_options, 
            compute_objective=self.compute_objective, stop_at=self.stop_at, 
            stop_when=self.stop_when, init=self.init, init_inv_cov=init_inv_cov)

        if self.return_history:
            self.precision_, self.covariance_, self.history_, self.n_iter_ = out
        else:
            self.precision_, self.covariance_, self.n_iter_ = out
        return self

    def fit(self, X, y):
        """Fit the TimeGraphicalLasso model to X."""
        self.classes_, n_samples = np.unique(y, return_counts=True)
        if X.ndim == 3:
            pass

        emp_cov = np.array(
            [
                empirical_covariance(
                    X[y == cl], assume_centered=self.assume_centered)
                for cl in self.classes_
            ])

        a = self._fit(emp_cov, n_samples)
        return a


class MissNet:
    """
    MISSNET: Mining of Switching Sparse Networks for Missing Value Imputation
    
    Parameters
    ----------
    alpha : float, default=0.5
        Trade-off parameter controlling the contribution of contextual matrix
        and time-series. If alpha = 0, network is ignored.
    beta : float, default=0.1
        Regularization parameter for sparsity.
    L : int, default=10
        Hidden dimension size.
    n_cl : int, default=1
        Number of clusters.
    """
    
    def __init__(self, alpha=0.5, beta=0.1, L=10, n_cl=1):
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.n_cl = n_cl

    def initialize(self, X, random_init=False):
        """Initialize model parameters."""
        # Given dataset
        self.T = X.shape[0]  # time
        self.N = X.shape[1]  # dim

        # Initialize model parameters
        self.init_network(X)

        if random_init:
            self.U = [np.random.rand(self.N, self.L) for _ in range(self.n_cl)]
            self.B = np.random.rand(self.L, self.L)
            self.z0 = np.random.rand(self.L)
            self.psi0 = np.random.rand(self.L, self.L)
            # noise
            self.sgmZ = np.random.rand()
            self.sgmX = [np.random.rand() for _ in range(self.n_cl)]
            self.sgmS = [np.random.rand() for _ in range(self.n_cl)]
            self.sgmV = [np.random.rand() for _ in range(self.n_cl)]
        else:
            self.U = [np.eye(self.N, self.L) for _ in range(self.n_cl)]
            self.B = np.eye(self.L)
            self.z0 = np.zeros(self.L)
            self.psi0 = np.eye(self.L)
            # noise
            self.sgmZ = 1.
            self.sgmX = [1. for _ in range(self.n_cl)]
            self.sgmS = [1. for _ in range(self.n_cl)]
            self.sgmV = [1. for _ in range(self.n_cl)]

        # Workspace
        # Forward algorithm
        self.mu_tt = [np.zeros((self.T, self.L)) for _ in range(self.n_cl)]
        self.psi_tt = [np.zeros((self.T, self.L, self.L)) for _ in range(self.n_cl)]
        self.mu_ = np.zeros((self.T, self.L))
        self.psi = np.zeros((self.T, self.L, self.L))
        self.I = np.eye(self.L)
        self.P = np.zeros((self.T, self.L, self.L))

        # Backward algorithm
        self.J = np.zeros((self.T, self.L, self.L))
        self.zt = np.zeros((self.T, self.L))
        self.ztt = np.zeros((self.T, self.L, self.L))
        self.zt1t = np.zeros((self.T, self.L, self.L))
        self.mu_h = np.zeros((self.T, self.L))
        self.psih = np.zeros((self.T, self.L, self.L))

        # M-step
        self.v = [np.zeros((self.N, self.L)) for _ in range(self.n_cl)]
        self.vv = [np.zeros((self.N, self.L, self.L)) for _ in range(self.n_cl)]

    def init_network(self, X):
        """Initialize network and cluster assignments."""
        X_interpolate = interpolate_matrix(X)
        self.F = np.random.choice(range(self.n_cl), self.T)
        self.update_MC()

        # initialize GGM (network)
        self.H = [np.zeros((self.N, self.N)) for _ in range(self.n_cl)]
        self.G = [np.zeros(self.N) for _ in range(self.n_cl)]
        self.S = [np.zeros((self.N, self.N)) for _ in range(self.n_cl)]
        for k in range(self.n_cl):
            F_k = np.where(self.F == k)[0]
            test = TVGL(alpha=self.beta, beta=0, max_iter=1000, psi='laplacian', assume_centered=False)
            test.fit(X_interpolate[F_k], np.zeros(X_interpolate[F_k].shape[0]))
            self.H[k] = test.precision_[0]
            self.G[k] = np.nanmean(X_interpolate[F_k], axis=0)
            self.S[k] = self.normalize_precision(test.precision_[0])

    def normalize_precision(self, H):
        """Calculate partial correlation."""
        N, N = H.shape
        S = np.zeros(H.shape)
        for i in range(N):
            for j in range(N):
                if i == j:
                    S[i, j] = 1
                else:
                    S[i, j] = -(H[i, j] / (np.sqrt(H[i, i]) * np.sqrt(H[j, j])))
        return S

    def fit(self, X, random_init=False, max_iter=20, min_iter=3, tol=5, verbose=True, savedir='./temp'):
        """EM algorithm for training the model."""
        make_dir(savedir, delete=True)

        W = ~np.isnan(X)
        if verbose:
            print('\nnumber of nan', np.count_nonzero(np.isnan(X)), 'percentage', 
                  np.round(np.count_nonzero(np.isnan(X))/X.size*100, decimals=1), '%\n')
        self.initialize(X, random_init)
        history = {'lle': [], 'time': []}
        min_lle, tol_counter = np.inf, 0
        
        for iteration in range(1, max_iter + 1):
            tic = time.time()
            try:
                """ E-step """
                # infer F, Z
                lle = self.forward_viterbi(X, W)
                if verbose and self.n_cl > 1: 
                    print('\tcluster assignments', [np.count_nonzero(self.F == k) for k in range(self.n_cl)])
                self.backward()
                # infer V
                self.update_latent_context()

                """ M-step """
                # update parameters
                lle -= self.solve_model(X, W, return_loglikelihood=True)

                """ Update the missing values and networks"""
                self.update_networks(X, W)  # update G,H,S

                toc = time.time()
                history['time'].append(toc - tic)
                history['lle'].append(lle)
                if verbose: 
                    print(f'\titer= {iteration}, lle= {lle:.3f}, time= {toc-tic:.3f} [sec]')

                # if LLE no more reduce then convergence (tol=5)
                if iteration > min_iter:
                    if lle < min_lle:
                        min_lle = lle
                        self.save_pkl(savedir)  # save the best model
                        tol_counter = 0
                    else:
                        tol_counter += 1

                    if tol_counter >= tol:  # Early stopping (end of training)
                        self.load_pkl(savedir)  # load the best model
                        return history
            except:
                if verbose: print("EM algorithm Error\n")
                self.load_pkl(savedir)  # load the best model
                return history

        message = "the EM algorithm did not converge\n"
        message += "Consider increasing 'max_iter'"
        warnings.warn(message)
        self.load_pkl(savedir)  # load the best model

        return history

    def forward_viterbi(self, X, W):
        """Forward algorithm and Viterbi approximation."""
        J = np.zeros((self.n_cl, self.T))
        F = np.zeros((self.n_cl, self.T), dtype=int)
        J_tt1 = np.zeros((self.n_cl, self.n_cl, self.T))
        K = [[[[] for _ in range(self.T)] for _ in range(self.n_cl)] for _ in range(self.n_cl)]
        P_ = [np.zeros((self.T, self.L, self.L)) for _ in range(self.n_cl)]
        mu_tt = np.zeros((self.n_cl, self.n_cl, self.T, self.L))
        psi_tt = np.zeros((self.n_cl, self.n_cl, self.T, self.L, self.L))
        mu_tt1 = np.zeros((self.n_cl, self.n_cl, self.T, self.L))
        psi_tt1 = np.zeros((self.n_cl, self.n_cl, self.T, self.L, self.L))

        for t in range(self.T):
            for i in range(self.n_cl):
                for j in range(self.n_cl):
                    lle = 0
                    ot = W[t]  # observed dim
                    xt = X[t, ot]  # observed data
                    It = np.eye(xt.shape[0])
                    Ht = self.U[i][ot, :]  # observed object latent matrix

                    if t == 0:
                        psi_tt1[i, j, 0] = self.psi0
                        mu_tt1[i, j, 0] = self.z0
                    else:
                        psi_tt1[i, j, t] = self.B @ self.psi_tt[j][t-1] @ self.B.T + self.sgmZ * self.I
                        mu_tt1[i, j, t] = self.B @ self.mu_tt[j][t-1]

                    delta = xt - Ht @ mu_tt1[i, j, t]
                    sigma = Ht @ psi_tt1[i, j, t] @ Ht.T + self.sgmX[i] * It
                    inv_sigma = np.linalg.pinv(sigma)

                    K[i][j][t] = psi_tt1[i, j, t] @ Ht.T @ inv_sigma
                    mu_tt[i, j, t] = mu_tt1[i, j, t] + K[i][j][t] @ delta
                    psi_tt[i, j, t] = (self.I - K[i][j][t] @ Ht) @ psi_tt1[i, j, t]

                    # Kalman LLE
                    df = delta @ inv_sigma @ delta / 2
                    sign, logdet = np.linalg.slogdet(inv_sigma)
                    lle -= self.L / 2 * np.log(2 * np.pi)
                    lle += sign * logdet / 2 - df

                    J_tt1[i, j, t] -= lle
                    J_tt1[i, j, t] -= self.window_Gaussian_LLE(X, W, self.U[i], mu_tt[i, j], self.G[i], self.H[i], t)
                    J_tt1[i, j, t] -= np.log(self.A[i, j]) if self.A[i, j] > 0 else np.log(0.01)

                if t > 0:
                    j_min = np.argmin(J_tt1[i, :, t] + J[:, t-1])
                    J[i, t] = J_tt1[i, j_min, t] + J[j_min, t-1]
                else:
                    j_min = np.argmin(J_tt1[i, :, t] + 0)
                    J[i, t] = J_tt1[i, j_min, t] + 0
                F[i, t] = j_min
                self.psi_tt[i][t] = psi_tt[i, j_min, t]
                self.mu_tt[i][t] = mu_tt[i, j_min, t]
                P_[i][t] = psi_tt1[i, j_min, t]

        i_min = np.argmin(J[:, self.T-1])
        self.F = F[i_min, :]
        self.F[0] = self.F[1]  # first point tends to be different
        self.update_MC()
        for t in range(self.T):
            f_k = self.F[t]
            self.psi[t] = self.psi_tt[f_k][t]
            self.mu_[t] = self.mu_tt[f_k][t]
            self.P[t] = P_[f_k][t]

        return J[i_min][self.T-1]

    def window_Gaussian_LLE(self, X, W, U, mu_tt, mu, invcov, t, window=1):
        """Compute window Gaussian log-likelihood."""
        if t == 0:
            return self.Gaussian_LLE(np.nan_to_num(X[t]) + (1-W[t])*(U@mu_tt[t]), mu, invcov)

        st = 0 if t < window else t-window
        infe = np.array([U@mutt for mutt in mu_tt[st:t]])
        return self.Gaussian_LLE(np.nan_to_num(X[st:t]) + (1-W[st:t])*infe, mu, invcov)

    def Gaussian_LLE(self, x, mu, invcov):
        """Compute Gaussian log-likelihood."""
        if x.ndim == 1:
            N, P = 1, x.shape[0]
            det = np.linalg.det(invcov) if np.linalg.det(invcov) > 0 else 1
            lle = -np.log(2*np.pi)*(N*P/2)
            lle += np.log(det)*(N/2) - (np.linalg.multi_dot([x-mu, invcov, (x-mu).T]))/2

        if x.ndim == 2:
            N, P = x.shape
            det = np.linalg.det(invcov) if np.linalg.det(invcov) > 0 else 1
            lle = -np.log(2*np.pi)*(N*P/2)
            lle += np.log(det)*(N/2) - np.sum(np.diag(np.linalg.multi_dot([x-mu, invcov, (x-mu).T])))/2
        return lle

    def update_MC(self):
        """Update Markov process."""
        S = np.zeros((self.n_cl, 1))
        SS = np.zeros((self.n_cl, self.n_cl))
        for t in range(1, self.T):
            e = np.zeros((self.n_cl, 1))
            e[self.F[t]] = 1

            e1 = np.zeros((self.n_cl, 1))
            e1[self.F[t-1]] = 1

            S += e
            SS += e @ e1.T
        S = S[:, 0]
        SS = SS.T

        # A[i,j]: probability of transitions from j(t-1) to i(t)
        self.A = SS @ np.linalg.inv(np.diag(S))
        self.A0 = np.zeros(self.n_cl)
        self.A0[self.F[0]] = 1

    def backward(self):
        """Backward algorithm."""
        self.mu_h[-1] = self.mu_[-1]
        self.psih[-1] = self.psi[-1]

        for t in reversed(range(self.T - 1)):
            self.J[t] = self.psi[t] @ self.B.T @ np.linalg.pinv(self.P[t+1])
            self.psih[t] = self.psi[t] + self.J[t] @ (self.psih[t+1] - self.P[t+1]) @ self.J[t].T
            self.mu_h[t] = self.mu_[t] + self.J[t] @ (self.mu_h[t+1] - self.B @ self.mu_[t])
        self.zt[:] = self.mu_h[:]

        for t in range(self.T):
            if t > 0:
                self.zt1t[t] = self.psih[t] @ self.J[t-1].T
                self.zt1t[t] += np.outer(self.mu_h[t], self.mu_h[t-1])
            self.ztt[t] = self.psih[t] + np.outer(self.mu_h[t], self.mu_h[t])

    def update_latent_context(self):
        """Bayes' theorem."""
        for k in range(self.n_cl):
            Minv = np.linalg.pinv(self.U[k].T @ self.U[k] + self.sgmS[k] / self.sgmV[k] * np.eye(self.L))
            gamma = self.sgmS[k] * Minv
            for i in range(self.N):
                self.v[k][i] = Minv @ self.U[k].T @ self.S[k][i, :]
                self.vv[k][i] = gamma + np.outer(self.v[k][i], self.v[k][i])

    def solve_model(self, X, W, return_loglikelihood=False):
        """Update parameters."""
        self.z0 = self.zt[0]
        self.psi0 = self.ztt[0] - np.outer(self.zt[0], self.zt[0])
        self.update_transition_matrix()
        self.update_contextual_covariance()
        self.update_transition_covariance()
        lle = self.update_object_latent_matrix(X, W, return_loglikelihood)
        self.update_network_covariance()
        self.update_observation_covariance(X, W)

        return lle

    def update_transition_matrix(self):
        """Update transition matrix."""
        self.B = sum(self.zt1t) @ np.linalg.pinv(sum(self.ztt))

    def update_contextual_covariance(self):
        """Update contextual covariance."""
        for k in range(self.n_cl):
            val = sum(np.trace(self.vv[k][i]) for i in range(self.N)) / (self.N * self.L)
            self.sgmV[k] = sum(np.trace(self.vv[k][i]) for i in range(self.N)) / (self.N * self.L)

    def update_transition_covariance(self):
        """Update transition covariance."""
        val = np.trace(
            sum(self.ztt[1:])
            - sum(self.zt1t[1:]) @ self.B.T
            - (sum(self.zt1t[1:]) @ self.B.T).T
            + self.B @ sum(self.ztt[:-1]) @ self.B.T
        )
        self.sgmZ = val / ((self.T - 1) * self.L)

    def update_object_latent_matrix(self, X, W, return_loglikelihood=False):
        """Update object latent matrix."""
        lle = 0
        for k in range(self.n_cl):
            F_k = np.where(self.F == k)[0]
            F_k = F_k[np.where(F_k != 0)]
            if len(F_k) == 0: 
                continue
            for i in range(self.N):
                A1 = self.alpha / self.sgmS[k] * sum(
                    self.S[k][i, j] * self.v[k][j] for j in range(self.N))
                A1 += (1 - self.alpha) / self.sgmX[k] * sum(
                    np.nan_to_num(W[t, i] * X[t, i] * self.zt[t]) for t in F_k)
                A2 = self.alpha / self.sgmS[k] * sum(self.vv[k])
                A2 += (1 - self.alpha) / self.sgmX[k] * sum(
                    W[t, i] * self.ztt[t] for t in F_k)
                self.U[k][i, :] = A1 @ np.linalg.pinv(A2)

            if return_loglikelihood:
                for i in range(self.N):
                    delta = self.S[k][i] - self.U[k][i] @ self.v[k][i]
                    sigma = self.sgmS[k] * np.eye(self.N) + self.U[k] @ self.vv[k][i] @ self.U[k].T
                    inv_sigma = np.linalg.pinv(sigma)
                    sign, logdet = np.linalg.slogdet(inv_sigma)
                    lle -= self.L / 2 * np.log(2 * np.pi)
                    lle += sign * logdet / 2 - delta @ inv_sigma @ delta / 2
        return lle

    def update_network_covariance(self):
        """Update network covariance."""
        for k in range(self.n_cl):
            val = sum(self.S[k][i].T @ self.S[k][i] - 2 * self.S[k][i] @ (self.U[k] @ self.v[k][i]) for i in range(self.N))
            val += np.trace(self.U[k] @ sum(self.vv[k]) @ self.U[k].T)
            self.sgmS[k] = val / self.N ** 2

    def update_observation_covariance(self, X, W):
        """Update observation covariance."""
        for k in range(self.n_cl):
            val = 0
            F_k = np.where(self.F == k)[0]
            F_k = F_k[np.where(F_k != 0)]
            if len(F_k) == 0: 
                continue
            for t in F_k:
                ot = W[t, :]
                xt = X[t, ot]
                Ht = self.U[k][ot, :]
                val += np.trace(Ht @ self.ztt[t] @ Ht.T)
                val += xt @ xt - 2 * xt @ (Ht @ self.zt[t])
            self.sgmX[k] = val / W[F_k, :].sum()

    def update_networks(self, X, W):
        """Impute X and Update network S via graphical lasso."""
        for k in range(self.n_cl):
            F_k = np.where(self.F == k)[0]
            if len(F_k) == 0: 
                continue
            Y = np.array([self.U[k] @ zt for zt in self.zt])
            X_impute = np.nan_to_num(X) + (1-W)*Y
            test = TVGL(alpha=self.beta, beta=0, max_iter=1000, psi='laplacian', assume_centered=False)
            test.fit(X_impute[F_k], np.zeros(X_impute[F_k].shape[0]))
            self.H[k] = test.precision_[0]
            self.G[k] = np.nanmean(X_impute[F_k], axis=0)
            self.S[k] = self.normalize_precision(test.precision_[0])

    def imputation(self):
        """Generate imputed data."""
        X_impute = np.zeros((self.T, self.N))
        for k in range(self.n_cl):
            F_k = np.where(self.F == k)[0]
            if len(F_k) == 0: 
                continue
            Z_temp = np.array([self.U[k] @ zt for zt in self.zt])
            X_impute[F_k] = Z_temp[F_k]
        return X_impute

    def save_pkl(self, outdir):
        """Save model to pickle file."""
        with open(f'{outdir}/model.pkl', mode='wb') as f:
            pickle.dump(self, f)

    def load_pkl(self, outdir):
        """Load model from pickle file."""
        if os.path.isfile(f'{outdir}/model.pkl'):
            with open(f'{outdir}/model.pkl', mode='rb') as f:
                model = pickle.load(f)

            self.U = model.U
            self.B = model.B
            self.z0 = model.z0
            self.psi0 = model.psi0
            self.sgmZ = model.sgmZ
            self.sgmX = model.sgmX
            self.sgmS = model.sgmS
            self.sgmV = model.sgmV

            self.F = model.F
            self.H = model.H
            self.G = model.G
            self.S = model.S

            # Forward algorithm
            self.mu_tt = model.mu_tt
            self.psi_tt = model.psi_tt
            self.mu_ = model.mu_
            self.psi = model.psi
            self.I = model.I
            self.P = model.P

            # Backward algorithm
            self.J = model.J
            self.zt = model.zt
            self.ztt = model.ztt
            self.zt1t = model.zt1t
            self.mu_h = model.mu_h
            self.psih = model.psih

            # M-step
            self.v = model.v
            self.vv = model.vv


def missnet_impute(incomp_data, alpha=0.5, beta=0.1, L=10, n_cl=1, max_iteration=20, 
                   tol=5, random_init=False, verbose=True):
    """
    Perform imputation using the MISSNET algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with missing values represented as NaNs.
    alpha : float, default=0.5
        Trade-off parameter controlling the contribution of contextual matrix
        and time-series. If alpha = 0, network is ignored.
    beta : float, default=0.1
        Regularization parameter for sparsity.
    L : int, default=10
        Hidden dimension size.
    n_cl : int, default=1
        Number of clusters.
    max_iteration : int, default=20
        Maximum number of iterations for convergence.
    tol : int, default=5
        Tolerance for early stopping criteria.
    random_init : bool, default=False
        Whether to use random initialization for latent variables.
    verbose : bool, default=True
        Whether to display progress information.

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.
    """
    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)

    if verbose:
        print(f"(IMPUTATION) MISS NET")
        print(f"\tMatrix Shape: {incomp_data.shape[0]}, {incomp_data.shape[1]}")
        print(f"\talpha: {alpha}")
        print(f"\tbeta: {beta}")
        print(f"\tL: {L}")
        print(f"\tn_cl: {n_cl}")
        print(f"\tmax_iteration: {max_iteration}")

    start_time = time.time()

    missnet_model = MissNet(alpha=alpha, beta=beta, L=L, n_cl=n_cl)
    missnet_model.fit(incomp_data, random_init=random_init, max_iter=max_iteration, 
                     tol=tol, verbose=verbose)
    recov_data = missnet_model.imputation()

    end_time = time.time()

    recov[m_mask] = recov_data[m_mask]

    if verbose:
        print(f"\n> logs: imputation miss_net - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov


if __name__ == "__main__":
    # Example usage
    print("MISSNET Imputation Script")
    print("=" * 50)
    
    # Create sample data with missing values
    np.random.seed(42)
    n_timesteps, n_features = 100, 10
    data = np.random.randn(n_timesteps, n_features)
    
    # Introduce missing values (20% missing)
    mask = np.random.random(data.shape) < 0.2
    data_with_missing = data.copy()
    data_with_missing[mask] = np.nan
    
    print(f"Original data shape: {data.shape}")
    print(f"Missing values: {np.isnan(data_with_missing).sum()}/{data_with_missing.size}")
    print(f"Missing percentage: {np.isnan(data_with_missing).sum()/data_with_missing.size*100:.1f}%")
    
    # Perform imputation
    imputed_data = missnet_impute(
        data_with_missing,
        alpha=0.5,
        beta=0.1,
        L=10,
        n_cl=1,
        max_iteration=20,
        verbose=True
    )
    
    # Calculate reconstruction error
    mse = np.nanmean((data[mask] - imputed_data[mask]) ** 2)
    mae = np.nanmean(np.abs(data[mask] - imputed_data[mask]))
    
    print(f"\nImputation Results:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"All missing values imputed: {not np.any(np.isnan(imputed_data[mask]))}")
