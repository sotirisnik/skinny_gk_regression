import numpy as np
import scipy.stats as stats
from numba import njit
import pandas as pd
import json
import time

def ols_regression(X, y):
    """
    Standard Ordinary Least Squares (OLS): beta = (X^T * X)^-1 * X^T * y
    """
    #Compute normal equations
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)
    
    #try to solve the system
    try:
        beta_ols = np.linalg.solve(XtX, Xty)
        return beta_ols
    except np.linalg.LinAlgError:
        #If the matrix is singular, e.g., p > n, then use pseudo-inverse
        return np.linalg.pinv(X) @ y

@njit(fastmath=True)#It allows simd to optimize cpu
def single_step_reorthogonalization(V, pk, k, zeta_min):
    """
    Algorithm 3: Single-step Reorthogonalization.
    Follows the nested if/else structure from our paper.
    """
    norm_pk_initial = np.linalg.norm(pk)
    
    #Require pk: if ||pk||2 == 0 then
    if norm_pk_initial < 1e-18:
        return 0.0, np.zeros_like(pk) # βk = 0, v_{k+1} = 0
    else:
        #First Step
        V_k = np.ascontiguousarray( V[:, :k] )
        
        h1 = V_k.T @ pk#np.dot(V[:, :k].T, pk)
        pk_1 = pk - V_k @ h1#pk - np.dot(V[:, :k], h1)
        norm_pk_1 = np.linalg.norm(pk_1)

        #if ||pk_1||2 >= zeta_min * ||pk||2 then
        if norm_pk_1 >= zeta_min * norm_pk_initial:
            beta_k = norm_pk_1
            return beta_k, pk_1 / beta_k#βk = ||pk_1||2, v_{k+1} = pk_1 / βk
        else:
            #βk = 0, v_{k+1} = 0
            return 0.0, np.zeros_like(pk)

@njit(fastmath=True)#It allows simd to optimize cpu
def two_step_reorthogonalization(V, pk, k, zeta_min):
    """
    Algorithm 3: Two-step Reorthogonalization.
    Follows the nested if/else structure from our paper.
    """
    norm_pk_initial = np.linalg.norm(pk)
    
    #Require pk: if ||pk||2 == 0 then
    if norm_pk_initial < 1e-18:
        return 0.0, np.zeros_like(pk) # βk = 0, v_{k+1} = 0
    else:
        #First Step
        V_k = np.ascontiguousarray( V[:, :k] )
        
        h1 = V_k.T @ pk#np.dot(V[:, :k].T, pk)
        pk_1 = pk - V_k @ h1#pk - np.dot(V[:, :k], h1)
        norm_pk_1 = np.linalg.norm(pk_1)

        #if ||pk_1||2 >= zeta_min * ||pk||2 then
        if norm_pk_1 >= zeta_min * norm_pk_initial:
            beta_k = norm_pk_1
            return beta_k, pk_1 / beta_k # βk = ||pk_1||2, v_{k+1} = pk_1 / βk
        else:
            #Second Step
            h2 = V_k.T @ pk_1#np.dot(V[:, :k].T, pk_1)
            pk_2 = pk_1 - V_k @ h2#pk_1 - np.dot(V[:, :k], h2)
            norm_pk_2 = np.linalg.norm(pk_2)
            
            #if ||pk_2||2 >= zeta_min * ||pk_1||2 then
            if norm_pk_2 >= zeta_min * norm_pk_1:
                beta_k = norm_pk_2
                return beta_k, pk_2 / beta_k
            else:
                #else: βk = 0, v_{k+1} = 0
                return 0.0, np.zeros_like(pk)

@njit(fastmath=True)
def skinny_gk_regression(X, y):
    """
    Skinny Golub-Kahan Regression with:
    a) Initialization of the first vector of matrix V according to Section 2.3,
    b) 2-step reorthogonalization from Algorithm Algorithm 3,
    c) Solve the bidiagonal system from Algorithm 2
    """
    n, p = X.shape
    
    U = np.asfortranarray( np.zeros((n, p)) )
    V = np.asfortranarray( np.zeros((p, p)) )

    zeta_min = np.sqrt(0.8)#0.8944271909999159 

    #Diagonal elements of matrix B1
    alphas = np.zeros(p)#Main diagonal
    betas = np.zeros(p-1)#Super-diagonal
    
    #Select initial vector
    v1 = y @ X
    V[ :, 0 ] = v1 / np.linalg.norm(v1)

    #First step of bidiagonalization (k=0)
    #r_k = X * v1 (n x 1) -> corresponds to line r_k = X*vk of Algorithm 1, with β_{κ-1} = 0
    r_0 = X @ np.ascontiguousarray( V[ :, 0] )#np.dot( X, V[ :, 0 ] )
    alphas[0] = np.linalg.norm(r_0)
    U[ :, 0 ] = r_0 / alphas[0]
    
    #Iterative bidiagonalization process
    for k in range(1, p):        
        #Right residual vector
        p_k = (X.T @ U[ :, k-1 ]) - (alphas[k-1] * V[ :, k-1 ])#np.dot( X.T, U[ :, k-1 ] ) - alphas[k-1] * V[ :, k-1 ]
        
        beta_val, v_next = two_step_reorthogonalization(V, p_k, k, zeta_min)

        #Invariant subspace detected
        if beta_val < 1e-18:
            break

        betas[k-1] = beta_val
        V[ :, k ] = v_next
        
        #Left residual vector
        rk = ( X @ V[ :, k ] ) - (betas[k-1] * U[ :, k-1 ])#np.dot(X, V[ :, k ] ) - betas[k-1] * U[ :, k-1 ]
        alphas[k] = np.linalg.norm(rk)
        
        #Rank deficiency detected
        if alphas[k] < 1e-18:
            break
            
        U[ :, k ] = rk / alphas[k]

    k_final = 0
    for i in range(p):
        if alphas[i] > 1e-15:#Tolerance
            k_final += 1
        else:
            break
                
    #Construct the bidiagonal matrix B1
    B1 = np.diag(alphas) + np.diag(betas, k=1)
    
    U_sub = U[:, :k_final]
    B_sub = B1[:k_final, :k_final]
    V_sub = V[:, :k_final]

    #Solve the upper bidiagonal system B1*w = U^T*y with back-substitution
    rhs = np.ascontiguousarray(U_sub.T) @ y
    w = np.zeros(k_final)#w = np.zeros(p)
    w[k_final-1] = rhs[k_final-1] / alphas[k_final-1]
    for i in range(k_final-2,-1,-1):
        w[i] = (rhs[i] - betas[i] * w[i+1]) / alphas[i]

    #Compute \hat{beta}
    beta_hat = np.ascontiguousarray( V_sub ) @ w
    
    return beta_hat, B_sub, U_sub, V_sub

def smart_skinny_gk_regression(X, y):
    m, p = X.shape
    if m >= 5000:
        return skinny_gk_regression.py_func(X, y)
    return skinny_gk_regression(X, y)

@njit(fastmath=True)
def skinny_gk_regression_single_reortho(X, y):
    """
    Skinny Golub-Kahan Regression with:
    a) Initialization of the first vector of matrix V according to Section 2.3,
    b) Single-step reorthogonalization from Algorithm 3,
    c) Solve the bidiagonal system from Algorithm 2.
    """
    n, p = X.shape
    
    U = np.asfortranarray( np.zeros((n, p)) )
    V = np.asfortranarray( np.zeros((p, p)) )

    zeta_min = np.sqrt(0.8)#0.8944271909999159 

    #Diagonal elements of matrix B1
    alphas = np.zeros(p)#Main diagonal
    betas = np.zeros(p-1)#Super-diagonal
    
    #Select initial vector v1
    v1 = y @ X
    V[ :, 0 ] = v1 / np.linalg.norm(v1)

    #First step of bidiagonalization (k=0)
    #r_k = X * v1 (n x 1) -> corresponds to line r_k = X*vk of Algorithm 1, with β_{κ-1} = 0
    r_0 = X @ np.ascontiguousarray( V[ :, 0] )
    alphas[0] = np.linalg.norm(r_0)
    U[ :, 0 ] = r_0 / alphas[0]
    
    #Iterative bidiagonalization process
    for k in range(1, p):
        #Right residual vector
        p_k = (X.T @ U[ :, k-1 ]) - (alphas[k-1] * V[ :, k-1 ])#np.dot( X.T, U[ :, k-1 ] ) - alphas[k-1] * V[ :, k-1 ]
        
        beta_val, v_next = single_step_reorthogonalization(V, p_k, k, zeta_min)

        #Invariant subspace detected
        if beta_val < 1e-18:
            break

        betas[k-1] = beta_val
        V[ :, k ] = v_next
        
        #Left residual vector
        rk = ( X @ V[ :, k ] ) - (betas[k-1] * U[ :, k-1 ])#np.dot(X, V[ :, k ] ) - betas[k-1] * U[ :, k-1 ]
        alphas[k] = np.linalg.norm(rk)
        
        #Rank deficiency detected
        if alphas[k] < 1e-18:
            break
            
        U[ :, k ] = rk / alphas[k]

    k_final = 0
    for i in range(p):
        if alphas[i] > 1e-15:#Tolerance
            k_final += 1
        else:
            break
                
    #Construct the bidiagonal matrix B1
    B1 = np.diag(alphas) + np.diag(betas, k=1)
    
    U_sub = U[:, :k_final]
    B_sub = B1[:k_final, :k_final]
    V_sub = V[:, :k_final]

    #Solve the upper bidiagonal system B1*w = U^T*y with back-substitution
    rhs = np.ascontiguousarray(U_sub.T) @ y
    w = np.zeros(k_final)
    w[k_final-1] = rhs[k_final-1] / alphas[k_final-1]
    for i in range(k_final-2,-1,-1):
        w[i] = (rhs[i] - betas[i] * w[i+1]) / alphas[i]

    #Compute \hat{beta}
    #beta_hat = V @ w#np.dot( V, w )
    beta_hat = np.ascontiguousarray( V_sub ) @ w
    
    return beta_hat, B_sub, U_sub, V_sub

@njit(fastmath=True)
def skinny_gk_regression_no_reortho(X, y):
    """
    Skinny Golub-Kahan Regression with:
    a) Initialization of the first vector of matrix V according to Section 2.3
    b) Without reorthogonalization,
    c) Solve the bidiagonal system from Algorithm 2.
    """
    n, p = X.shape
    
    U = np.asfortranarray( np.zeros((n, p)) )
    V = np.asfortranarray( np.zeros((p, p)) )

    zeta_min = np.sqrt(0.8)#0.8944271909999159 

    #Diagonal elements of matrix B1
    alphas = np.zeros(p)#Main diagonal
    betas = np.zeros(p-1)#Super-diagonal
    
    #Select initial vector v1
    v1 = y @ X
    V[ :, 0 ] = v1 / np.linalg.norm(v1)

    #First step of bidiagonalization (k=0)
    #r_k = X * v1 (n x 1) -> corresponds to line r_k = X*vk of Algorithm 1, with β_{κ-1} = 0
    r_0 = X @ np.ascontiguousarray( V[ :, 0] )#np.dot( X, V[ :, 0 ] )
    alphas[0] = np.linalg.norm(r_0)
    U[ :, 0 ] = r_0 / alphas[0]
    
    #Iterative bidiagonalization process
    for k in range(1, p):
        #Right residual vector
        p_k = (X.T @ U[ :, k-1 ]) - (alphas[k-1] * V[ :, k-1 ])#np.dot( X.T, U[ :, k-1 ] ) - alphas[k-1] * V[ :, k-1 ]
        beta_val = np.linalg.norm(p_k)

        #Invariant subspace detected
        if beta_val < 1e-18:
            break

        v_next = p_k / beta_val

        betas[k-1] = beta_val
        V[ :, k ] = v_next
        
        #Left residual vector
        rk = ( X @ V[ :, k ] ) - (betas[k-1] * U[ :, k-1 ])#np.dot(X, V[ :, k ] ) - betas[k-1] * U[ :, k-1 ]
        alphas[k] = np.linalg.norm(rk)
        
        #Rank deficiency detected
        if alphas[k] < 1e-18:
            break
            
        U[ :, k ] = rk / alphas[k]

    k_final = 0
    for i in range(p):
        if alphas[i] > 1e-15:#Tolerance
            k_final += 1
        else:
            break
                
    #Construct the bidiagonal matrix B1
    B1 = np.diag(alphas) + np.diag(betas, k=1)
    
    U_sub = U[:, :k_final]
    B_sub = B1[:k_final, :k_final]
    V_sub = V[:, :k_final]

    #Solve the upper bidiagonal system B1*w = U^T*y with back-substitution
    rhs = np.ascontiguousarray(U_sub.T) @ y #rhs = np.dot( U.T, y )
    w = np.zeros(k_final)#w = np.zeros(p)
    w[k_final-1] = rhs[k_final-1] / alphas[k_final-1]
    for i in range(k_final-2,-1,-1):
        w[i] = (rhs[i] - betas[i] * w[i+1]) / alphas[i]

    #Compute \hat{beta}
    beta_hat = np.ascontiguousarray( V_sub ) @ w
    
    return beta_hat, B_sub, U_sub, V_sub

def skinny_svd_regression(X, y, tol=1e-15):
    """
    Skinny SVD Regression for comparison against Skinny-GK
    """
    #Compute Skinny SVD: X = U * S * V^T
    #if full_matrices=False, then U has dimensions (n, p) instead of (n, n)
    U, s_vals, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T#SVD returns V transposed

    t = np.sum( s_vals > tol )

    U_subset = U[ :, :t]
    s_subset = s_vals[ :t ]
    V_subset = V[ :, :t]

    #s is a vector containing the diagonal singular values of Sigma
    #The solution is: beta_hat = V * inv(Sigma) * U^T * y
    
    #Compute U_subset^T * y
    Ut_y = np.dot(U_subset.T, y)
    
    #Multiply by the inverse of diagonal matrix Sigma (simply 1 / s_i)
    w = Ut_y / s_subset
    
    #Multiply by V_subset
    beta_hat = np.dot(V_subset, w)
    
    return beta_hat, s_subset, V_subset