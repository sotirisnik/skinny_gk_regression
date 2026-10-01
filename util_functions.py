import numpy as np
import scipy.stats as stats
from numba import njit
import pandas as pd
import json
import time
from ucimlrepo import fetch_ucirepo
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes

def generate_data( m=1000, n=100, r=0.2, sparsity=33, rng=None, noise_multiplier=1.0 ):
    #m rows
    #n columns (features)
    #sparsity := number of active non-zero features

    #Construct covariance matrix
    #s_ij = r^|i-j|
    """Sigma = np.zeros( (n,n) )
    for i in range(n):
        for j in range(n):
            Sigma[i,j] = r ** abs(i-j)
    """
    ii, jj = np.indices((n, n))
    Sigma = r ** np.abs(ii - jj)

    #Sample (m x n) matrix X ~ N(0, Sigma)
    X = rng.multivariate_normal( np.zeros(n),
                                    Sigma,
                                    size=m )

    #Construct true sparse regression vector beta (n x 1)
    beta_true = np.zeros(n)
    beta_true[:sparsity] = 1.0
    rng.shuffle(beta_true)

    #Compute: y = dot(X,beta_true) + noise
    sigma = 1/3.0 * np.sqrt(sparsity/m)
    if noise_multiplier != 1.0:
        sigma *= noise_multiplier
    noise = rng.normal(0, sigma, size=m)
    y = np.dot(X, beta_true) + noise

    return X, y, beta_true

def handle_center(is_centered=True):
    if is_centered:#In case y was centered manually, e.g. real dataset
        return 1
    return 0#In case y was already with zero-mean, e.g. Simulation

def variable_selection_gk(X, y, beta_hat, B1, V, alpha=0.05, is_centered=True):
    """
    Implementation of the F-test for variable selection according to Section 2.4
    We compute the statistical significance of each beta_hat coefficient.
    """
    n, p = X.shape
    
    k_final = B1.shape[0]
    dof = n - k_final - handle_center(is_centered)#degrees of freedom

    #We compute the sum of squared errors
    residuals = y - np.dot( X, beta_hat )
    sum_squares = np.sum(residuals**2) / (dof)
    
    #We compute the diagonal elements g_ii, of the covariance matrix (X^TX)^-1)
    #Ο υπολογισμός γίνεται χωρίς χρήση της np.linalg.inv μέσω της ανάλυσης GK
    
    #We solve the system B1^T * Z = V.T with forward substitution to compute Z since B1^T is lower triangular
    Z = np.linalg.solve(B1.T, V.T)
    
    #The g_ii are obtained from the sum of squares of the columns of Z
    g_ii = np.sum(Z**2, axis=0)
    
    #We compute the F_i statistic for each variable
    f_stats = (beta_hat**2) / (sum_squares * g_ii)

    #We find the critical value through the Fisher distribution
    #The ppf function gives us this critical value that variables must exceed to be considered significant
    #The arguments are: the confidence level 1-alpha, and the degrees of freedom (1, dof)
    critical_value = stats.f.ppf(1 - alpha, 1, dof)
    
    #A variable is selected if its F_i exceeds the critical value
    significant = f_stats > critical_value

    return f_stats, significant, critical_value

def variable_selection_svd(X, y, beta_hat, s_vals, V, alpha=0.05, tol=1e-15, is_centered=True):
    """
    F-test for SVD using the already computed matrices.
    """
    n, p = X.shape
    
    #Determine Rank from the already existing s_vals
    t = np.sum(s_vals > tol)
    s_subset = s_vals[:t]
    V_subset = V[:, :t]
    
    #Computation of S^2 (Residual Variance)
    residuals = y - np.dot(X, beta_hat)
    dof = n - t - handle_center(is_centered)
    sum_squares = np.sum(residuals**2) / (dof)
    
    #Computation of g_ii (Diagonal of the inverse): sum( (V_ij / s_j)^2 )
    g_ii = np.sum((V_subset / s_subset)**2, axis=1)
    
    #Computation of F-statistics
    f_stats = (beta_hat**2) / (sum_squares * g_ii)
    critical_value = stats.f.ppf(1 - alpha, 1, dof)
    significant = f_stats > critical_value
    
    return f_stats, significant, critical_value

def variable_selection_ols(X, y, beta_hat, alpha=0.05):
    """
    F-test for OLS using precomputed beta_hat.
    """
    n, p = X.shape
    #Computation of S^2
    residuals = y - np.dot(X, beta_hat)
    sum_squares = np.sum(residuals**2) / (n - p)
    
    #Computation of g_ii from the diagonal of (X^T X)^-1
    XtX = np.dot(X.T, X)
    inv_XtX = np.linalg.inv(XtX)
    g_ii = np.diag(inv_XtX)
    
    #F-statistic
    f_stats = (beta_hat**2) / (sum_squares * g_ii)
    critical_value = stats.f.ppf(1 - alpha, 1, n - p)
    significant = f_stats > critical_value
    
    return f_stats, significant, critical_value

def prepare_data(dataset_name):
    """
    Prepares the Real Datasets for Regression.
    Selects the target y and applies the transformations (e.g.,sqrt).
    """

    ids = {'glass': 42, 'wine': 186, 'breast': 17, 'magic': 159, 'music': 862}

    if dataset_name in ['diamond', 'california', 'diabetes']:
        pass
    elif dataset_name == 'music':
        df = pd.read_csv( 'datasets/Acoustic Features.csv')
    else:
        #For the remaining 4 that are supported by the ucimlrepo API
        ids = {'glass': 42, 'wine': 186, 'breast': 17, 'magic': 159}
        repo = fetch_ucirepo(id=ids[dataset_name])
        df = pd.concat([repo.data.features, repo.data.targets], axis=1)

    X = None
    y = None
    
    if dataset_name == 'glass':
        df_encoded = pd.get_dummies(df, columns=['Type_of_glass'], drop_first=True)
        keys_to_drop = ['Al']#, 'Type_of_glass']
        X    = df_encoded.drop(columns=keys_to_drop).values#Remove y and classification label       
        keys = df_encoded.drop(columns=keys_to_drop).columns.values
        #Target_y = Aluminum (Al).
        y = df_encoded['Al'].values
        y = np.sqrt(y)#sqrt(Al)
    elif dataset_name == 'wine':
        keys_to_drop = ['density']
        X    = df.drop(columns=keys_to_drop).values
        keys = df.drop(columns=keys_to_drop).columns.values
        #Target_y = Density
        y = df['density'].values
    elif dataset_name == 'breast':
        df_encoded = pd.get_dummies(df, columns=['Diagnosis'], drop_first=True)
        keys_to_drop = ['smoothness3']
        X    = df_encoded.drop(columns=keys_to_drop).values
        keys = df_encoded.drop(columns=keys_to_drop).columns.values
        y = df_encoded['smoothness3'].values#Target_y = Smoothness3
    elif dataset_name == 'music':
        df_encoded = pd.get_dummies(df, columns=['Class'], drop_first=True)
        keys_to_drop = ['_MFCC_Mean_1']
        X    = df_encoded.drop(columns=keys_to_drop).values
        keys = df_encoded.drop(columns=keys_to_drop).columns.values
        y = df_encoded['_MFCC_Mean_1'].values
    elif dataset_name == 'magic':
        df_encoded = pd.get_dummies(df, columns=['class'], drop_first=True)
        #Target: sqrt(fConc1)
        keys_to_drop = ['fConc1']
        X   = df_encoded.drop(columns=keys_to_drop).values
        keys = df_encoded.drop(columns=keys_to_drop).columns.values
        y = df_encoded['fConc1'].values
        y = np.sqrt(y)
    elif dataset_name == 'diamond':
        df = sns.load_dataset('diamonds')

        #Convert categorical variables to Dummy Variables (One-Hot Encoding)
        #This will create new columns like cut_Ideal, color_E etc.
        df_encoded = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)

        #Separate Target (price) and Features
        keys_to_drop = ['price']
        X = df_encoded.drop(columns=keys_to_drop).values
        
        #Convert keys to Numpy Array so your indexing works
        keys = df_encoded.drop(columns=keys_to_drop).columns.to_numpy()
        
        y = df_encoded['price'].values.astype(float)
        y = np.log1p(y)
    elif dataset_name == 'california':
        data = fetch_california_housing()
        X = data.data
        keys = np.array(data.feature_names)
        y = data.target#Target: House value (in $100k)
    elif dataset_name == 'diabetes':
        data = load_diabetes()
        X = data.data
        keys = np.array(data.feature_names)
        y = data.target#Target: Disease progression after one year
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return X, y, keys

def flatten_dict(d, parent_key='', sep='_'):
    #traverse dictionary and if the value contains another dictionary then we have to flatten it
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

#https://github.com/sotirisnik/PFLEGO
def GetMeanStd( x_ ):
    x_ = np.array( x_ ).copy().reshape( (-1,1) )
    mean_ = x_.mean()
    std_ = x_.std()
    z = 1.96
    return mean_,  z * std_ / ( len(x_)**0.5 )